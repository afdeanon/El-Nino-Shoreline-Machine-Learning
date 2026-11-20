"""
Streamlit demo: El Niño and Southern California shoreline change.

Loads the models saved in models/locked_test/ and runs them live (nothing is trained here).

Run from the project folder:
    streamlit run scripts/streamlit_demo_app.py
"""
import json
import time
import warnings
from pathlib import Path

import __main__
import altair as alt
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")   # hide scikit-learn version warnings

ROOT_DIR   = Path(__file__).resolve().parent.parent
LOCKED_DIR = ROOT_DIR / "models" / "locked_test"
Q1_FOLDER  = LOCKED_DIR / "q1_last_4_winters"
Q2_FOLDER  = LOCKED_DIR / "q2_22_random_sections"
TARGET     = "shoreline_change_m"
MAX_TRAINING_RONI = 1.94     # strongest El Niño the Question 1 models were trained on (2015–16)

RONI_MODELS = ["Linear: beach + wave + RONI", "GAM: beach + wave + RONI",
               "RF: beach + wave + RONI", "GBT: beach + wave + RONI"]
STRONG_EL_NINO = 1.5         # NOAA calls an El Niño "strong" at about +1.5

# RONI and the coast-wide average shoreline change for every winter (from processed/final_modeling_dataset.csv)
WINTERS = pd.DataFrame({
    "winter": ["2000-2001", "2001-2002", "2002-2003", "2003-2004", "2004-2005", "2005-2006", "2006-2007",
               "2007-2008", "2008-2009", "2009-2010", "2010-2011", "2011-2012", "2012-2013", "2013-2014",
               "2014-2015", "2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020", "2020-2021"],
    "RONI": [-0.70, -0.16, 0.80, 0.18, 0.52, -0.76, 0.50, -1.56, -0.82, 1.20, -1.34,
             -0.78, -0.38, -0.28, 0.54, 1.94, -0.62, -1.02, 0.70, 0.14, -1.30],
    "average_change": [-9.447, 1.000, -7.943, -7.326, -7.439, -8.250, -7.478, -9.038, -5.245, -13.877, -7.281,
                       -6.637, -5.742, -9.695, -1.579, -8.155, -7.922, -4.492, -10.070, -1.810, -9.152],
})


# ---------------------------------------------------------------- custom model classes
# The baseline and "transect average + ..." models were saved with these classes (copied from the notebook),
# so Python needs them to load those files.

class TransectAverage(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        training = pd.DataFrame({"transect_id": X["transect_id"].values, "change": np.asarray(y)})
        self.averages_ = training.groupby("transect_id")["change"].mean()
        self.overall_ = training["change"].mean()
        return self

    def predict(self, X):
        predictions = X["transect_id"].map(self.averages_)
        predictions = predictions.fillna(self.overall_)
        return predictions.to_numpy()


class TransectAveragePlus(BaseEstimator, RegressorMixin):
    def __init__(self, model=None, features=None):
        self.model = model
        self.features = features

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.means_ = y.groupby(X["transect_id"]).mean()
        self.overall_ = float(y.mean())
        leftovers = y - X["transect_id"].map(self.means_)
        self.model_ = clone(self.model).fit(X[self.features], leftovers)
        return self

    def predict(self, X):
        normal = X["transect_id"].map(self.means_).fillna(self.overall_).to_numpy()
        return normal + self.model_.predict(X[self.features])


__main__.TransectAverage = TransectAverage          # pickle looks for the classes in __main__
__main__.TransectAveragePlus = TransectAveragePlus


# ---------------------------------------------------------------- loading (cached, so it only happens once)

@st.cache_data
def load_table(path):
    return pd.read_csv(path)


@st.cache_data
def load_info(folder):
    with open(folder / "model_info.json") as f:
        return json.load(f)


@st.cache_data
def load_results(path, test_set):
    results = pd.read_csv(path)
    if "test set" in results.columns:
        results = results[results["test set"] == test_set]
    return results.set_index("model")


@st.cache_resource
def load_models(folder):
    """Load every saved model in a folder: {model name: model}."""
    info = load_info(folder)
    models = {}
    for file_name in info:
        if file_name.endswith(".pkl"):
            models[info[file_name]["model"]] = joblib.load(folder / file_name)
    return models


def features_of(info):
    """{model name: list of features it uses}."""
    return {info[f]["model"]: info[f]["features"] for f in info if f.endswith(".pkl")}


# ---------------------------------------------------------------- running the models live

def predict_live(models, features, rows, model_names=None):
    """Run saved models on some rows. Returns a table with one column per model."""
    if model_names is None:
        model_names = list(models)
    predictions = pd.DataFrame(index=rows.index)
    for name in model_names:
        predictions[name] = models[name].predict(rows[features[name]])
    return predictions


def run_the_test(models, features, rows, saved_results):
    """Predict every locked test row with every model and score it. Returns the live scores."""
    start = time.perf_counter()
    predictions = predict_live(models, features, rows)
    seconds = time.perf_counter() - start
    table = pd.DataFrame({
        "Live R²": [r2_score(rows[TARGET], predictions[m]) for m in predictions],
        "Live MAE (m)": [mean_absolute_error(rows[TARGET], predictions[m]) for m in predictions],
    }, index=predictions.columns)
    table["Notebook R²"] = saved_results.loc[table.index, "Test R²"]
    table["Matches notebook"] = np.where((table["Live R²"] - table["Notebook R²"]).abs() < 0.001, "✓", "✗")   # ±0.001
    return {"table": table, "seconds": seconds, "models": len(models), "rows": len(rows)}


def show_test_results(run, saved_results, baseline_name, with_roni_chart):
    """Charts built from the live run (locked test) next to the notebook's cross-validation scores."""
    st.success(f"Ran {run['models']} saved models on {run['rows']:,} locked test rows in {run['seconds']:.1f} seconds.")
    scores = pd.DataFrame({"CV R²": saved_results.loc[run["table"].index, "CV R²"],
                           "Test R²": run["table"]["Live R²"]})
    if with_roni_chart:
        left, right = st.columns([3, 2])
    else:
        left, right = st.container(), None
    with left:
        st.markdown("**Scores: locked test (live) vs cross-validation (from the notebook)**")
        st.altair_chart(score_chart(scores, baseline_name), use_container_width=True)
    if right is not None:
        with right:
            st.markdown("**Does adding RONI help?**")
            st.altair_chart(roni_effect_chart(scores), use_container_width=True)
    with st.expander("Details: live scores vs the notebook"):
        st.dataframe(run["table"].round(3), use_container_width=True)


COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def compass(degrees):
    return COMPASS[int((degrees % 360) / 22.5 + 0.5) % 16]


def beach_inputs_table(row, all_beaches, include_roni=False):
    """The model inputs for one beach, in readable form, compared with every beach on the coast."""
    r = row.iloc[0]
    shore = np.degrees(np.arctan2(r["shore_orientation_sin"], r["shore_orientation_cos"])) % 360
    waves = np.degrees(np.arctan2(r["wave_dir_sin"], r["wave_dir_cos"])) % 360

    def compared(column):
        value = r[column]
        if pd.isna(value):
            return "missing (the models fill in the median)"
        share = (all_beaches[column].dropna() < value).mean() * 100
        return f"higher than {share:.0f}% of beaches"

    slope_text = "missing" if pd.isna(r["beach_slope"]) else f"{r['beach_slope']:.3f}"
    table = pd.DataFrame({
        "Value": [slope_text, f"{r['BeachWidth_m']:.0f} m", f"{shore:.0f}° ({compass(shore)})",
                  f"{r['winter_wave_power']:.2f}", f"from {waves:.0f}° ({compass(waves)})"],
        "Compared with all beaches": [compared("beach_slope"), compared("BeachWidth_m"), "direction the beach faces",
                                      compared("winter_wave_power"), "direction the waves come from"],
    }, index=["Beach slope", "Beach width", "Shore faces", "Average winter wave power (CDIP)", "Average wave direction"])
    if include_roni:
        table.loc["RONI (this winter)"] = [f"{r['RONI']:+.2f}", "El Niño > 0 > La Niña"]
    return table


# ---------------------------------------------------------------- colors and legends

# One color per model, used in every chart, so GAM is always the same blue, GBT always teal, and so on
MODEL_COLORS = {"Baseline": "#9ca3af", "Linear": "#f472b6", "GAM": "#38bdf8",
                "RF": "#a78bfa", "Random forest": "#a78bfa", "GBT": "#2dd4bf", "Actual": "#ef4444"}


def model_color(label):
    """The color for a model label, e.g. "GAM: beach + wave" or "Baseline (overall average)"."""
    return next((color for name, color in MODEL_COLORS.items() if label.startswith(name)), "#e5e7eb")


def model_color_scale(labels):
    """Altair color scale that gives each model its color from MODEL_COLORS."""
    return alt.Scale(domain=list(labels), range=[model_color(label) for label in labels])


def legend(items):
    """Color-coded legend above a map. items = [(color, label, shape)], shape is "line" or "box"."""
    parts = []
    for color, label, shape in items:
        size = "width:34px;height:5px;" if shape == "line" else "width:20px;height:20px;"
        parts.append(f'<span style="display:inline-flex;align-items:center;margin-right:28px;">'
                     f'<span style="{size}background:{color};border-radius:3px;margin-right:9px;"></span>{label}</span>')
    st.markdown(f'<div style="font-size:1.05rem;margin:2px 0 10px 0;">{"".join(parts)}</div>', unsafe_allow_html=True)


def gradient_legend(left_label, colors, right_label, middle_label):
    """Color bar legend above a map: left_label [gradient] right_label, with a label under the middle."""
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:14px;font-size:1.05rem;margin:2px 0 2px 0;">'
        f'<span>{left_label}</span>'
        f'<div style="width:320px;height:18px;border-radius:4px;background:linear-gradient(to right, {", ".join(colors)});"></div>'
        f'<span>{right_label}</span></div>'
        f'<div style="font-size:0.9rem;color:#9ca3af;margin:0 0 10px 0;">{middle_label}</div>', unsafe_allow_html=True)


def takeaway(text):
    """Key takeaway box at the top of a tab."""
    st.markdown(f'<div style="border-left:6px solid #38bdf8;background:rgba(56,189,248,0.10);padding:14px 18px;'
                f'border-radius:6px;font-size:1.15rem;margin:6px 0 18px 0;"><b>Key takeaway:</b> {text}</div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------- charts (Altair: matches the app's theme, hover for values)

def big_text(chart):
    """Larger fonts for every chart (Altair's defaults are small). Labels are kept inside the chart, not cut off."""
    return (chart.properties(autosize=alt.AutoSizeParams(type="fit", contains="padding"))
                 .configure_axis(labelFontSize=15, titleFontSize=16, titlePadding=12, labelLimit=0)
                 .configure_legend(labelFontSize=15, symbolSize=200, labelLimit=0)
                 .configure_title(fontSize=19, anchor="start"))


def score_chart(results, baseline_name):
    """Cross-validation vs locked-test R² for every model."""
    data = results[["CV R²", "Test R²"]].rename(columns={"CV R²": "Cross-validation (notebook)",
                                                         "Test R²": "Locked test (live)"})
    data = data.rename_axis("model").reset_index().melt(id_vars="model", var_name="score", value_name="R2")
    model_order = list(results.index)
    bars = alt.Chart(data).mark_bar().encode(
        y=alt.Y("model:N", sort=model_order, title=None, axis=alt.Axis(labelLimit=400)),
        yOffset=alt.YOffset("score:N"),
        x=alt.X("R2:Q", title="R² (higher is better)"),
        color=alt.Color("score:N", title=None, legend=alt.Legend(orient="top")),
        tooltip=["model:N", "score:N", alt.Tooltip("R2:Q", title="R²", format=".3f")],
    )
    baseline_value = pd.DataFrame({"R2": [results.loc[baseline_name, "Test R²"]], "label": [f"{baseline_name} (test)"]})
    baseline = alt.Chart(baseline_value).mark_rule(color="gray", strokeDash=[6, 4], size=2).encode(
        x="R2:Q", tooltip=[alt.Tooltip("label:N", title="Baseline to beat"), alt.Tooltip("R2:Q", title="R²", format=".3f")],
    )
    return big_text((bars + baseline).properties(height=36 * len(model_order)))


def roni_effect_chart(results):
    """Change in R² from adding RONI, for each model type."""
    test = results["Test R²"]
    cv = results["CV R²"]
    pairs = {
        "Transect average":       ("Baseline: transect average", "Transect average + linear RONI"),
        "Transect average + GBT": ("Transect average + GBT beach + wave (control)", "Transect average + GBT beach + wave + RONI"),
        "Linear":                 ("Linear: beach + wave", "Linear: beach + wave + RONI"),
        "GAM":                    ("GAM: beach + wave", "GAM: beach + wave + RONI"),
        "Random forest":          ("RF: beach + wave", "RF: beach + wave + RONI"),
        "GBT":                    ("GBT: beach + wave", "GBT: beach + wave + RONI"),
    }
    rows = []
    for family, (without_roni, with_roni) in pairs.items():
        rows.append({"model": family, "score": "Cross-validation (notebook)", "change": cv[with_roni] - cv[without_roni]})
        rows.append({"model": family, "score": "Locked test (live)", "change": test[with_roni] - test[without_roni]})
    bars = alt.Chart(pd.DataFrame(rows)).mark_bar().encode(
        y=alt.Y("model:N", sort=list(pairs), title=None, axis=alt.Axis(labelLimit=400)),
        yOffset=alt.YOffset("score:N"),
        x=alt.X("change:Q", title="Change in R² from adding RONI (negative = worse)"),
        color=alt.Color("score:N", title=None, legend=alt.Legend(orient="top")),
        tooltip=["model:N", "score:N", alt.Tooltip("change:Q", title="Change in R²", format="+.3f")],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="gray").encode(x="x:Q")
    return big_text((bars + zero).properties(height=320))


def what_if_chart(row, chosen_roni, models, features, baseline_prediction):
    """Each RONI model's prediction for one beach as RONI changes; the real winter is the red dot."""
    roni_values = np.round(np.arange(-2.0, 2.51, 0.05), 2)
    lines = []
    for name in RONI_MODELS:
        X = pd.concat([row[features[name]]] * len(roni_values), ignore_index=True)
        X["RONI"] = roni_values                                  # only RONI changes
        lines.append(pd.DataFrame({"RONI": roni_values, "prediction": models[name].predict(X), "model": name.split(":")[0]}))
    lines = pd.concat(lines, ignore_index=True)
    model_lines = alt.Chart(lines).mark_line(strokeWidth=2.5).encode(
        x=alt.X("RONI:Q", title="RONI (negative = La Niña, positive = El Niño)"),
        y=alt.Y("prediction:Q", title="Predicted shoreline change (m)", scale=alt.Scale(zero=False)),
        color=alt.Color("model:N", title=None, legend=alt.Legend(orient="top"), scale=model_color_scale(["Linear", "GAM", "RF", "GBT"])),
        tooltip=["model:N", alt.Tooltip("RONI:Q", format="+.2f"), alt.Tooltip("prediction:Q", title="Predicted change (m)", format=".2f")],
    )
    baseline = alt.Chart(pd.DataFrame({"y": [baseline_prediction]})).mark_rule(color="gray", strokeDash=[6, 4], size=2).encode(
        y="y:Q", tooltip=[alt.Tooltip("y:Q", title="Baseline: transect average (m)", format=".2f")])
    chosen = alt.Chart(pd.DataFrame({"x": [chosen_roni]})).mark_rule(color="gray", size=3, opacity=0.6).encode(
        x="x:Q", tooltip=[alt.Tooltip("x:Q", title="Chosen RONI", format="+.1f")])
    actual = pd.DataFrame({"RONI": [row["RONI"].iloc[0]], "change": [row[TARGET].iloc[0]]})
    actual_dot = alt.Chart(actual).mark_point(color="red", filled=True, size=200).encode(
        x="RONI:Q", y="change:Q",
        tooltip=[alt.Tooltip("RONI:Q", title="Actual RONI", format="+.2f"),
                 alt.Tooltip("change:Q", title="Actual change (m)", format=".2f")])
    return big_text((model_lines + baseline + chosen + actual_dot).properties(height=420))


def beach_cross_section(actual_change, predictions, title, x_domain=None):
    """Side view of the beach: shoreline before winter (0), where it actually ended up, and each model's prediction.
    Left = toward land (erosion), right = toward the ocean."""
    rows = [{"label": "Actual", "position": actual_change}]
    for name, value in predictions.items():
        rows.append({"label": name, "position": value})
    points = pd.DataFrame(rows)
    points["text"] = points["position"].map(lambda v: f"{v:+.1f} m")
    row_order = list(points["label"])

    if x_domain is None:
        left_edge  = min(points["position"].min(), 0) - 8
        right_edge = max(points["position"].max(), 0) + 8
    else:
        left_edge, right_edge = x_domain
    x_scale = alt.Scale(domain=[left_edge, right_edge])

    sand = alt.Chart(pd.DataFrame({"x": [left_edge], "x2": [actual_change]})).mark_rect(color="#d8c08f", opacity=0.55).encode(
        x=alt.X("x:Q", scale=x_scale), x2="x2:Q")
    ocean = alt.Chart(pd.DataFrame({"x": [actual_change], "x2": [right_edge]})).mark_rect(color="#3b82c4", opacity=0.45).encode(
        x=alt.X("x:Q", scale=x_scale), x2="x2:Q")
    changed = alt.Chart(pd.DataFrame({"x": [min(0, actual_change)], "x2": [max(0, actual_change)]})).mark_rect(
        color="#f5e6c8", opacity=0.35).encode(x=alt.X("x:Q", scale=x_scale), x2="x2:Q")
    before = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="white", strokeDash=[6, 4], size=2).encode(
        x=alt.X("x:Q", scale=x_scale), tooltip=[alt.Tooltip("x:Q", title="Shoreline before winter (m)")])
    after = alt.Chart(pd.DataFrame({"x": [actual_change]})).mark_rule(color="red", size=3).encode(
        x=alt.X("x:Q", scale=x_scale), tooltip=[alt.Tooltip("x:Q", title="Actual shoreline after winter (m)", format="+.2f")])
    dots = alt.Chart(points).mark_point(filled=True, size=220, opacity=1).encode(
        x=alt.X("position:Q", scale=x_scale,
                title="← toward land (erosion)        shoreline position after winter, relative to before (m)        toward ocean (gain) →"),
        y=alt.Y("label:N", sort=row_order, title=None, axis=alt.Axis(labelLimit=300)),
        color=alt.Color("label:N", scale=model_color_scale(row_order), legend=None),
        stroke=alt.value("white"), strokeWidth=alt.value(1.5),
        tooltip=["label:N", alt.Tooltip("position:Q", title="Shoreline change (m)", format="+.2f")],
    )
    labels = alt.Chart(points).mark_text(align="left", dx=12, fontSize=16, fontWeight="bold", color="white").encode(
        x=alt.X("position:Q", scale=x_scale), y=alt.Y("label:N", sort=row_order), text="text:N")
    return big_text((sand + ocean + changed + before + after + dots + labels).properties(title=title, height=56 * len(row_order) + 40))


def coast_average_chart(coast_rows, models, features, chosen_roni):
    """Average predicted change along the whole coast, as RONI changes, for each RONI model."""
    roni_values = np.round(np.arange(-2.0, 3.01, 0.25), 2)
    rows = []
    for roni in roni_values:
        X = coast_rows.copy()
        X["RONI"] = roni
        for name in RONI_MODELS:
            rows.append({"RONI": roni, "model": name.split(":")[0],
                         "average": models[name].predict(X[features[name]]).mean()})
    data = pd.DataFrame(rows)
    outside = alt.Chart(pd.DataFrame({"x": [MAX_TRAINING_RONI], "x2": [3.0]})).mark_rect(color="gray", opacity=0.2).encode(
        x="x:Q", x2="x2:Q", tooltip=[alt.Tooltip("x:Q", title="Outside training data above RONI")])
    model_lines = alt.Chart(data).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X("RONI:Q", title="RONI (negative = La Niña, positive = El Niño)"),
        y=alt.Y("average:Q", title="Average predicted change, whole coast (m)", scale=alt.Scale(zero=False)),
        color=alt.Color("model:N", title=None, legend=alt.Legend(orient="top"), scale=model_color_scale(["Linear", "GAM", "RF", "GBT"])),
        tooltip=["model:N", alt.Tooltip("RONI:Q", format="+.2f"), alt.Tooltip("average:Q", title="Average change (m)", format=".2f")],
    )
    chosen = alt.Chart(pd.DataFrame({"x": [chosen_roni]})).mark_rule(color="white", size=2, opacity=0.7).encode(x="x:Q")
    return big_text((outside + model_lines + chosen).properties(height=380))


def feature_importance_chart(importance):
    """Drop-column importance for GAM and GBT: how much the error grows when each feature is removed."""
    order = list(importance[["GAM", "GBT"]].mean(axis=1).sort_values(ascending=False).index)
    data = importance.reset_index().melt(id_vars="feature", var_name="model", value_name="increase")
    bars = alt.Chart(data).mark_bar().encode(
        y=alt.Y("feature:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
        yOffset=alt.YOffset("model:N"),
        x=alt.X("increase:Q", title="Increase in error (MAE, m) when the feature is removed"),
        color=alt.Color("model:N", title=None, legend=alt.Legend(orient="top"), scale=model_color_scale(["GAM", "GBT"])),
        tooltip=["feature:N", "model:N", alt.Tooltip("increase:Q", title="Error increase (m)", format="+.3f")],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="gray").encode(x="x:Q")
    return big_text((bars + zero).properties(height=42 * len(order)))


def test_winters_chart(winters, locked_winters):
    """Every winter's RONI and coast-wide average change: training winters in gray, locked test winters in orange."""
    data = winters.copy()
    data["set"] = np.where(data["winter"].isin(locked_winters), "Locked test winter", "Training winter")
    test_roni = data.loc[data["set"] == "Locked test winter", "RONI"]
    x_scale = alt.Scale(domain=[-2, 2.2])
    test_range = alt.Chart(pd.DataFrame({"x": [test_roni.min()], "x2": [test_roni.max()]})).mark_rect(
        color="#f59e0b", opacity=0.15).encode(x=alt.X("x:Q", scale=x_scale), x2="x2:Q",
                                              tooltip=[alt.Tooltip("x:Q", title="Test RONI from", format="+.2f"),
                                                       alt.Tooltip("x2:Q", title="Test RONI to", format="+.2f")])
    strong = alt.Chart(pd.DataFrame({"x": [STRONG_EL_NINO], "x2": [2.2]})).mark_rect(color="#ef4444", opacity=0.12).encode(
        x=alt.X("x:Q", scale=x_scale), x2="x2:Q", tooltip=[alt.Tooltip("x:Q", title="Strong El Niño above RONI", format="+.1f")])
    dots = alt.Chart(data).mark_circle(size=220, opacity=1).encode(
        x=alt.X("RONI:Q", scale=x_scale, title="RONI (negative = La Niña, positive = El Niño)"),
        y=alt.Y("average_change:Q", title="Average shoreline change, whole coast (m)"),
        color=alt.Color("set:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=["Training winter", "Locked test winter"], range=["#9ca3af", "#f59e0b"])),
        tooltip=["winter:N", "set:N", alt.Tooltip("RONI:Q", format="+.2f"),
                 alt.Tooltip("average_change:Q", title="Average change (m)", format=".1f")],
    )
    labels = alt.Chart(data).mark_text(align="left", dx=10, fontSize=13, color="#d1d5db").encode(
        x=alt.X("RONI:Q", scale=x_scale), y="average_change:Q", text="winter:N")
    return big_text((test_range + strong + dots + labels).properties(height=380))


def predicted_vs_actual_chart(per_beach, model_name):
    """Each unseen beach: its average actual change vs the model's average prediction."""
    low = min(per_beach["actual"].min(), per_beach["predicted"].min()) - 2
    high = max(per_beach["actual"].max(), per_beach["predicted"].max()) + 2
    scale = alt.Scale(domain=[low, high])
    perfect = alt.Chart(pd.DataFrame({"x": [low, high], "y": [low, high]})).mark_line(
        color="white", strokeDash=[6, 4], opacity=0.7).encode(x=alt.X("x:Q", scale=scale), y=alt.Y("y:Q", scale=scale))
    dots = alt.Chart(per_beach).mark_circle(size=45, opacity=0.7,
                                            color=model_color(model_name)).encode(
        x=alt.X("actual:Q", scale=scale, title="Actual change, average over winters (m)"),
        y=alt.Y("predicted:Q", scale=scale, title=f"Predicted change (m), {model_name.split(':')[0]}"),
        tooltip=["transect_id:N", alt.Tooltip("actual:Q", title="Actual (m)", format="+.1f"),
                 alt.Tooltip("predicted:Q", title="Predicted (m)", format="+.1f")],
    )
    return big_text((perfect + dots).properties(height=440))


def sweep_chart(sweep, x_title, chosen_x, actual_x, actual_change, baseline_value):
    """Each model's prediction for one beach as one input changes. Red dot = the beach's real input and real change."""
    data = sweep.rename_axis("x").reset_index().melt(id_vars="x", var_name="model", value_name="prediction")
    model_lines = alt.Chart(data).mark_line(strokeWidth=2.5).encode(
        x=alt.X("x:Q", title=x_title),
        y=alt.Y("prediction:Q", title="Predicted shoreline change (m)", scale=alt.Scale(zero=False)),
        color=alt.Color("model:N", title=None, legend=alt.Legend(orient="top"), scale=model_color_scale(list(sweep.columns))),
        tooltip=["model:N", alt.Tooltip("x:Q", title=x_title, format=".3f"),
                 alt.Tooltip("prediction:Q", title="Predicted change (m)", format=".2f")],
    )
    baseline = alt.Chart(pd.DataFrame({"y": [baseline_value]})).mark_rule(color="gray", strokeDash=[6, 4], size=2).encode(
        y="y:Q", tooltip=[alt.Tooltip("y:Q", title="Baseline (m)", format=".2f")])
    chosen = alt.Chart(pd.DataFrame({"x": [chosen_x]})).mark_rule(color="gray", size=3, opacity=0.6).encode(x="x:Q")
    layers = model_lines + baseline + chosen
    if not pd.isna(actual_x):
        actual_dot = alt.Chart(pd.DataFrame({"x": [actual_x], "y": [actual_change]})).mark_point(
            color="red", filled=True, size=200).encode(
            x="x:Q", y="y:Q", tooltip=[alt.Tooltip("x:Q", title="Real value", format=".3f"),
                                       alt.Tooltip("y:Q", title="Actual change (m)", format=".2f")])
        layers = layers + actual_dot
    return big_text(layers.properties(height=400))


# ---------------------------------------------------------------- maps

def change_color(value, scale):
    """Red for erosion, blue for gain, white near zero."""
    s = max(-1.0, min(1.0, value / scale))
    if s < 0:
        return [255, int(255 * (1 + s)), int(255 * (1 + s))]
    return [int(255 * (1 - s)), int(255 * (1 - s)), 255]


def move_point(lat, lon, seaward_bearing_deg, meters):
    """Move a point `meters` along the shore-normal direction (positive = toward the ocean, negative = toward land)."""
    bearing = np.radians(seaward_bearing_deg)
    new_lat = lat + meters * np.cos(bearing) / 111_320
    new_lon = lon + meters * np.sin(bearing) / (111_320 * np.cos(np.radians(lat)))
    return new_lat, new_lon


def shoreline_map(data, model_column, exaggeration):
    """One map with three shorelines: before winter (white), actual after winter (red), model's prediction (yellow)."""
    data = data.sort_values("transect_id").copy()
    data["actual_lat"], data["actual_lon"] = move_point(data["lat"], data["lon"], data["shore_orientation_deg"],
                                                        data["actual"] * exaggeration)
    data["pred_lat"], data["pred_lon"] = move_point(data["lat"], data["lon"], data["shore_orientation_deg"],
                                                    data["predicted"] * exaggeration)
    data["actual_text"] = data["actual"].map(lambda v: f"{v:+.1f} m")
    data["predicted_text"] = data["predicted"].map(lambda v: f"{v:+.1f} m")
    data["error_text"] = (data["predicted"] - data["actual"]).map(lambda v: f"{v:+.1f} m")

    lines = pd.DataFrame([
        {"name": "Before winter",       "color": [255, 255, 255], "path": data[["lon", "lat"]].values.tolist()},
        {"name": "Actual after winter", "color": [239, 68, 68],   "path": data[["actual_lon", "actual_lat"]].values.tolist()},
        {"name": "Predicted",           "color": [250, 204, 21],  "path": data[["pred_lon", "pred_lat"]].values.tolist()},
    ])
    line_layer = pdk.Layer("PathLayer", data=lines, get_path="path", get_color="color",
                           width_min_pixels=3, width_max_pixels=5, pickable=False)
    actual_dots = pdk.Layer("ScatterplotLayer", data=data, get_position="[actual_lon, actual_lat]",
                            get_fill_color=[239, 68, 68], get_radius=4, radius_min_pixels=3, pickable=True)
    predicted_dots = pdk.Layer("ScatterplotLayer", data=data, get_position="[pred_lon, pred_lat]",
                               get_fill_color=[250, 204, 21], get_radius=4, radius_min_pixels=3, pickable=True)
    view = pdk.data_utils.compute_view(data[["lon", "lat"]].values.tolist())
    view.zoom = min(view.zoom, 16)
    tooltip = {"html": "<b>{transect_id}</b><br/>Actual: {actual_text}<br/>Predicted: {predicted_text}<br/>Error: {error_text}"}
    return pdk.Deck(layers=[line_layer, actual_dots, predicted_dots], initial_view_state=view,
                    map_provider="carto", map_style=pdk.map_styles.CARTO_DARK, tooltip=tooltip)


def shoreline_map_section(rows, models, features, locations, model_name, key):
    """Run the chosen model live on a stretch of coast, then draw the three shorelines."""
    exaggeration = st.slider("Exaggerate the movement (×)", 1, 20, 10, key=key,
                             help="Real movements are 5–30 m, which is tiny on a map, so they're stretched to be visible.")
    start = time.perf_counter()
    predicted = predict_live(models, features, rows, [model_name])[model_name]
    seconds = time.perf_counter() - start
    data = rows[["transect_id", TARGET]].assign(predicted=predicted.values).rename(columns={TARGET: "actual"})
    data = data.merge(locations, on="transect_id")
    legend([("#ffffff", "Shoreline before winter", "line"), ("#ef4444", "Actual shoreline after winter", "line"),
            ("#facc15", f"Predicted by {model_name}", "line")])
    st.pydeck_chart(shoreline_map(data, model_name, exaggeration))
    st.caption(f"Ran **{model_name}** live on {len(rows)} transects ({seconds * 1000:.0f} ms). "
               f"Movement exaggerated {exaggeration}×. Where red and yellow overlap, the model was right. "
               "Hover a dot for values. Needs internet for the background map.")


def roni_help_by_beach(rows, models, features, family):
    """Run one model with and without RONI on every locked test row. For each beach: average change in error from adding RONI."""
    without_roni, with_roni = f"{family}: beach + wave", f"{family}: beach + wave + RONI"
    predictions = predict_live(models, features, rows, [without_roni, with_roni])
    error_without = (predictions[without_roni] - rows[TARGET]).abs()
    error_with = (predictions[with_roni] - rows[TARGET]).abs()
    per_row = pd.DataFrame({"transect_id": rows["transect_id"], "difference": error_with - error_without})
    return per_row.groupby("transect_id")["difference"].mean().reset_index()


def help_color(value, scale):
    """Green when RONI lowered the error, orange when it raised it, gray near zero."""
    s = min(1.0, abs(value) / scale)
    gray = np.array([120, 120, 120])
    target = np.array([34, 197, 94]) if value < 0 else np.array([249, 115, 22])
    return (gray + s * (target - gray)).astype(int).tolist()


def roni_help_map(per_beach, scale):
    """Every beach, colored by whether adding RONI helped (green) or hurt (orange)."""
    data = per_beach.copy()
    data["color"] = [help_color(v, scale) for v in data["difference"]]
    data["text"] = data["difference"].map(lambda v: f"{v:+.2f} m ({'helped' if v < 0 else 'hurt'})")
    layer = pdk.Layer("ScatterplotLayer", data=data, get_position="[lon, lat]", get_fill_color="color",
                      get_radius=120, radius_min_pixels=3, radius_max_pixels=10, pickable=True)
    view = pdk.data_utils.compute_view(data[["lon", "lat"]].values.tolist())
    tooltip = {"html": "<b>{transect_id}</b><br/>Change in error from adding RONI: {text}"}
    return pdk.Deck(layers=[layer], initial_view_state=view, map_provider="carto",
                    map_style=pdk.map_styles.CARTO_DARK, tooltip=tooltip)


def coast_forecast_map(coast, scale):
    """Every transect along the coast, colored by predicted change."""
    coast = coast.copy()
    coast["color"] = [change_color(v, scale) for v in coast["predicted"]]
    coast["predicted_text"] = coast["predicted"].map(lambda v: f"{v:+.1f} m")
    layer = pdk.Layer("ScatterplotLayer", data=coast, get_position="[lon, lat]", get_fill_color="color",
                      get_radius=120, radius_min_pixels=3, radius_max_pixels=10, pickable=True)
    view = pdk.data_utils.compute_view(coast[["lon", "lat"]].values.tolist())
    tooltip = {"html": "<b>{transect_id}</b><br/>Predicted change: {predicted_text}"}
    return pdk.Deck(layers=[layer], initial_view_state=view, map_provider="carto",
                    map_style=pdk.map_styles.CARTO_DARK, tooltip=tooltip)


# ---------------------------------------------------------------- page

st.set_page_config(page_title="El Niño & Shoreline Change", layout="wide")
st.markdown("""
<style>
html { font-size: 18px; }                                              /* everything scales up (default is 16px) */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { font-size: 1rem !important; }   /* captions */
[data-testid="stMetricValue"] { font-size: 2.4rem; }
button[data-baseweb="tab"] p { font-size: 1.15rem; }                   /* tab names */
#MainMenu, footer, .stDeployButton, [data-testid="stAppDeployButton"], [data-testid="stDecoration"] { visibility: hidden; }
</style>
""", unsafe_allow_html=True)
st.title("El Niño and Southern California Shoreline Change")
st.caption("Every model here was trained without the locked test data. Nothing is trained in this app: "
           "it loads the saved models and runs them live.")

locations = load_table(LOCKED_DIR / "transect_locations.csv")

tab1, tab2, tab3 = st.tabs(["Question 1: New winter", "Question 2: New coastline", "This winter (2026–27)"])


# ================================================================ Question 1
with tab1:
    st.header("Does El Niño (RONI) help predict a new winter?")
    q1_info     = load_info(Q1_FOLDER)
    q1_rows     = load_table(Q1_FOLDER / "test_rows.csv")
    q1_results  = load_results(LOCKED_DIR / "q1_results.csv", "B: last 4 winters")
    q1_models   = load_models(Q1_FOLDER)
    q1_features = features_of(q1_info)

    q1_test = q1_results["Test R²"]
    takeaway(f"Each beach's own history (the transect average, test R² {q1_test['Baseline: transect average']:.2f}) beats "
             f"every model. Adding RONI helped only the GAM, and only a little ({q1_test['GAM: beach + wave + RONI'] - q1_test['GAM: beach + wave']:+.3f} R²), "
             "while it made the tree models (RF, GBT) worse. RONI doesn't reliably help predict a new winter.")

    st.write(f"**Locked test winters:** {', '.join(q1_info['locked winters'])} · "
             f"models trained on the other 17 winters · {len(q1_rows):,} locked test rows")

    if st.button("▶ Run all saved models on the locked test winters", key="q1_run"):
        st.session_state["q1_live_results"] = run_the_test(q1_models, q1_features, q1_rows, q1_results)
    if "q1_live_results" in st.session_state:
        show_test_results(st.session_state["q1_live_results"], q1_results, "Baseline: transect average", with_roni_chart=True)
    else:
        st.info("Click **▶ Run** to test the saved models on the 4 locked winters. The results charts appear here.")

    # ---- did the test include a strong El Niño?
    st.subheader("Did the locked test include a strong El Niño?")
    st.altair_chart(test_winters_chart(WINTERS, q1_info["locked winters"]), use_container_width=True)
    test_roni = WINTERS.loc[WINTERS["winter"].isin(q1_info["locked winters"]), "RONI"]
    st.caption(f"Each dot is one winter. **Orange band** = RONI range of the locked test winters "
               f"({test_roni.min():+.2f} to {test_roni.max():+.2f}). **Red area** = strong El Niño (RONI above "
               f"+{STRONG_EL_NINO}). No test winter was a strong El Niño, so the test can't tell us how well RONI works "
               f"in the winters where it should matter most. Only one training winter (2015–16, {MAX_TRAINING_RONI:+.2f}) was. "
               "Also notice the biggest coast-wide erosion (2009–10) was a moderate El Niño, not the strongest.")

    # ---- where did adding RONI help?
    st.subheader("Where did adding RONI help?")
    family_names = {"GAM": "GAM", "Linear": "Linear", "RF": "Random forest", "GBT": "GBT"}
    help_family = st.selectbox("Model", list(family_names), format_func=family_names.get, key="q1_help_model")
    start = time.perf_counter()
    per_beach_help = roni_help_by_beach(q1_rows, q1_models, q1_features, help_family)
    seconds = time.perf_counter() - start
    helped_share = (per_beach_help["difference"] < 0).mean() * 100
    h1, h2 = st.columns(2)
    h1.metric("Beaches where adding RONI lowered the error", f"{helped_share:.0f}%")
    average_help = per_beach_help["difference"].mean()
    if abs(average_help) < 0.05:                     # under 5 cm: too small to call it helped or hurt
        verdict, verdict_color = "No real difference (under 5 cm)", "off"
    elif average_help < 0:
        verdict, verdict_color = "RONI helped", "normal"
    else:
        verdict, verdict_color = "RONI hurt", "inverse"
    h2.metric("Average change in error per beach", f"{average_help:+.2f} m", delta=verdict, delta_color=verdict_color)
    per_beach_help = per_beach_help.merge(locations, on="transect_id")
    help_scale = max(0.5, per_beach_help["difference"].abs().quantile(0.95))
    gradient_legend("RONI helped (lower error)", ["rgb(34,197,94)", "rgb(120,120,120)", "rgb(249,115,22)"],
                    "RONI hurt (higher error)", f"Gray = no real difference · strongest color = ±{help_scale:.1f} m")
    st.pydeck_chart(roni_help_map(per_beach_help, help_scale))
    st.caption(f"Ran {family_names[help_family]} with and without RONI live on all {len(q1_rows):,} locked test rows "
               f"({seconds:.1f} s). Each dot is one beach: the error with RONI minus the error without it, averaged over the "
               "4 test winters. If the colors look random, RONI isn't helping in any consistent place.")

    # ---- map
    st.subheader("Map: a stretch of coast in one locked winter")
    q1_sections = q1_rows.groupby(q1_rows["transect_id"].str[:11])["transect_id"].nunique()
    section_names = sorted(q1_sections.index)
    map_a, map_b, map_c = st.columns(3)
    with map_a:
        map_winter = st.selectbox("Winter", q1_info["locked winters"], key="q1_map_winter")
    with map_b:
        default_section = section_names.index(q1_sections[(q1_sections >= 15) & (q1_sections <= 40)].index.min())
        map_section = st.selectbox("Coastal section", section_names, index=default_section,
                                   format_func=lambda name: f"{name} ({q1_sections[name]} transects)", key="q1_map_section")
    with map_c:
        q1_names = list(q1_models)
        map_model = st.selectbox("Model", q1_names, index=q1_names.index("GBT: beach + wave + RONI"), key="q1_map_model")
    in_map = (q1_rows["winter"] == map_winter) & (q1_rows["transect_id"].str[:11] == map_section)
    shoreline_map_section(q1_rows[in_map], q1_models, q1_features, locations, map_model, "q1_exaggeration")

    # ---- one beach
    st.subheader("Pick a beach and a winter")
    col_a, col_b = st.columns(2)
    with col_a:
        transects = sorted(q1_rows["transect_id"].unique())
        winters_per_transect = q1_rows.groupby("transect_id")["winter"].nunique()
        first_complete = transects.index(winters_per_transect[winters_per_transect == 4].index.min())
        transect = st.selectbox("Transect", transects, index=first_complete, key="q1_transect")
    with col_b:
        winters_here = sorted(q1_rows.loc[q1_rows["transect_id"] == transect, "winter"].unique())
        winter = st.selectbox("Locked winter", winters_here, key="q1_winter")

    row = q1_rows[(q1_rows["transect_id"] == transect) & (q1_rows["winter"] == winter)]
    live = predict_live(q1_models, q1_features, row).iloc[0]            # every model, run live on this one row
    actual = row[TARGET].iloc[0]

    st.write(f"**Actual change:** {actual:.2f} m · **RONI that winter:** {row['RONI'].iloc[0]:+.2f}")
    all_beaches = q1_rows.drop_duplicates("transect_id")          # one row per beach, every beach on the coast
    with st.expander("This beach's inputs to the models"):
        st.dataframe(beach_inputs_table(row, all_beaches, include_roni=True), use_container_width=True)
    st.dataframe(pd.DataFrame({"Predicted change (m)": live.round(2),
                               "Error: predicted − actual (m)": (live - actual).round(2)}), use_container_width=True)

    st.subheader("The beach, side view")
    q1_model_set = st.radio("Show models", ["With RONI", "Without RONI"], horizontal=True, key="q1_model_set")
    if q1_model_set == "With RONI":
        shown = {"Baseline (transect average)": "Baseline: transect average",
                 "Linear": "Linear: beach + wave + RONI", "GAM": "GAM: beach + wave + RONI",
                 "Random forest": "RF: beach + wave + RONI", "GBT": "GBT: beach + wave + RONI"}
    else:
        shown = {"Baseline (transect average)": "Baseline: transect average",
                 "Linear": "Linear: beach + wave", "GAM": "GAM: beach + wave",
                 "Random forest": "RF: beach + wave", "GBT": "GBT: beach + wave"}
    st.altair_chart(beach_cross_section(actual, {label: live[column] for label, column in shown.items()},
                                        f"{transect}, winter {winter}"), use_container_width=True)
    st.caption("Dashed white line = shoreline before winter (Oct–Nov) · red line = where it actually was after winter "
               "(Mar–Apr) · each dot = where a model predicted it would be. Sand on the left, ocean on the right.")

    st.subheader("What if this winter had a different El Niño?")
    st.write("Drag RONI (or press **Play**) and watch each model's predicted shoreline move. "
             "Everything else about this beach stays the same. The red line is what actually happened "
             f"(RONI {row['RONI'].iloc[0]:+.2f}); other RONI values are hypothetical.")

    # Every RONI model's prediction for this beach at every RONI value from -2.0 to +2.5, run live
    roni_grid = np.round(np.arange(-2.0, 2.51, 0.1), 1)
    sweep_rows = pd.concat([row] * len(roni_grid), ignore_index=True)
    sweep_rows["RONI"] = roni_grid
    sweep = predict_live(q1_models, q1_features, sweep_rows, RONI_MODELS)
    sweep.index = roni_grid
    sweep.columns = [name.split(":")[0] for name in sweep.columns]
    baseline_here = live["Baseline: transect average"]

    everything = list(sweep.values.ravel()) + [actual, 0, baseline_here]
    x_domain = [min(everything) - 6, max(everything) + 6]          # fixed, so the dots visibly slide

    def predictions_at(roni):
        at = {"Baseline (transect average)": baseline_here}
        for model_name in sweep.columns:
            at[model_name] = sweep.loc[round(roni, 1), model_name]
        return at

    slider_col, play_col = st.columns([4, 1])
    with slider_col:
        chosen_roni = st.slider("RONI (negative = La Niña, positive = El Niño)", -2.0, 2.5,
                                float(round(row["RONI"].iloc[0], 1)), 0.1, key="q1_what_if_roni")
    with play_col:
        st.write("")
        play = st.button("▶ Play (sweep −2 → +2.5)", key="q1_play")

    frame = st.empty()
    if play:
        for roni in roni_grid:
            frame.altair_chart(beach_cross_section(actual, predictions_at(roni), f"RONI = {roni:+.1f}", x_domain),
                               use_container_width=True)
            time.sleep(0.12)
    else:
        frame.altair_chart(beach_cross_section(actual, predictions_at(chosen_roni), f"RONI = {chosen_roni:+.1f}",
                                               x_domain), use_container_width=True)
    st.caption("Dashed white line = shoreline before winter · red line = where it actually ended up that winter · "
               "each dot = where a model predicts it would end up at the chosen RONI. The baseline ignores RONI, so it never moves.")

    with st.expander("Full curves: each model's prediction across all RONI values"):
        st.altair_chart(what_if_chart(row, chosen_roni, q1_models, q1_features, baseline_here), use_container_width=True)
        st.caption("Gray dashed line = baseline (transect average) · gray vertical line = chosen RONI · "
                   "red dot = what actually happened. Hover for values.")


# ================================================================ Question 2
with tab2:
    st.header("Do beach and wave features explain shoreline change on new coastline?")
    q2_info     = load_info(Q2_FOLDER)
    q2_rows     = load_table(Q2_FOLDER / "test_rows.csv")
    q2_results  = load_results(LOCKED_DIR / "q2_results.csv", "A: 22 random sections")
    q2_models   = load_models(Q2_FOLDER)
    q2_features = features_of(q2_info)

    takeaway(f"Beach and wave features barely predict change on coastline the models never saw: the best model's test R² "
             f"is only {q2_results['Test R²'].max():.2f}. Beach slope helps a little, but every effect is small.")

    st.write(f"**Locked test sections:** {len(q2_info['locked sections'])} coastal sections "
             f"({q2_rows['transect_id'].nunique()} transects) · models trained on the other 88 sections")
    st.write("The baseline is the **overall average**: an unseen beach has no history of its own.")

    if st.button("▶ Run all saved models on the locked test sections", key="q2_run"):
        st.session_state["q2_live_results"] = run_the_test(q2_models, q2_features, q2_rows, q2_results)
    if "q2_live_results" in st.session_state:
        show_test_results(st.session_state["q2_live_results"], q2_results, "Baseline: overall average", with_roni_chart=False)
    else:
        st.info("Click **▶ Run** to test the saved models on the 22 locked sections. The results chart appears here.")

    # ---- feature importance (from the notebook, Section 23)
    st.subheader("Which features matter on unseen coastline?")
    importance_file = LOCKED_DIR / "q2_feature_importance.csv"
    if importance_file.exists():
        importance = load_table(importance_file).set_index("feature")
        st.altair_chart(feature_importance_chart(importance), use_container_width=True)
        both_useful = [f for f in importance.index if importance.loc[f, "GAM"] > 0 and importance.loc[f, "GBT"] > 0]
        st.caption("Drop-column importance from the notebook: retrain each model without one feature and measure how much "
                   "the error on unseen sections grows (10-fold cross-validation on the 88 training sections; the locked "
                   "test sections were not used). Positive = the feature helps; negative = the model does better without it. "
                   f"**Both models agree only on: {', '.join(both_useful)}.** Every effect is small (at most "
                   f"{importance.abs().max().max():.2f} m, on a typical error of about 7.5 m).")
    else:
        st.info("Run Section 23 of the notebook to create models/locked_test/q2_feature_importance.csv.")

    # ---- predicted vs actual
    st.subheader("Predicted vs actual on unseen beaches")
    q2_model_names = [name for name in q2_models if not name.startswith("Baseline")]
    scatter_model = st.selectbox("Model", q2_model_names, index=q2_model_names.index("GBT: beach + wave"),
                                 key="q2_scatter_model")
    scatter_predictions = predict_live(q2_models, q2_features, q2_rows, [scatter_model])[scatter_model]
    per_beach_q2 = pd.DataFrame({"transect_id": q2_rows["transect_id"], "actual": q2_rows[TARGET],
                                 "predicted": scatter_predictions}).groupby("transect_id").mean().reset_index()
    st.altair_chart(predicted_vs_actual_chart(per_beach_q2, scatter_model), use_container_width=True)
    st.caption(f"Ran {scatter_model} live. Each dot is one unseen beach, averaged over its winters. "
               "**Dashed line** = perfect predictions. If the model understood what makes a beach erode, the dots would "
               "follow the line. Instead they form a flat band: the predictions spread only about "
               f"±{per_beach_q2['predicted'].std():.1f} m while real beaches spread about ±{per_beach_q2['actual'].std():.1f} m "
               f"(R² on these beach averages = {r2_score(per_beach_q2['actual'], per_beach_q2['predicted']):.2f}).")

    # ---- map
    st.subheader("Map: a stretch of coast the models never saw")
    q2_sections = q2_rows.groupby(q2_rows["transect_id"].str[:11])["transect_id"].nunique()
    q2_section_names = sorted(q2_sections.index)
    map2_a, map2_b, map2_c = st.columns(3)
    with map2_a:
        default_section2 = q2_section_names.index(q2_sections[(q2_sections >= 15) & (q2_sections <= 40)].index.min())
        map2_section = st.selectbox("Coastal section (never seen)", q2_section_names, index=default_section2,
                                    format_func=lambda name: f"{name} ({q2_sections[name]} transects)", key="q2_map_section")
    with map2_b:
        winters_in_section = sorted(q2_rows.loc[q2_rows["transect_id"].str[:11] == map2_section, "winter"].unique())
        map2_winter = st.selectbox("Winter", winters_in_section, index=len(winters_in_section) - 1, key="q2_map_winter")
    with map2_c:
        q2_names = list(q2_models)
        map2_model = st.selectbox("Model", q2_names, index=q2_names.index("GBT: beach + wave"), key="q2_map_model")
    in_map2 = (q2_rows["winter"] == map2_winter) & (q2_rows["transect_id"].str[:11] == map2_section)
    shoreline_map_section(q2_rows[in_map2], q2_models, q2_features, locations, map2_model, "q2_exaggeration")

    # ---- one beach
    st.subheader("Pick an unseen beach and a winter")
    col_a, col_b = st.columns(2)
    with col_a:
        transect2 = st.selectbox("Transect (never seen by the models)", sorted(q2_rows["transect_id"].unique()),
                                 key="q2_transect")
    with col_b:
        winters_here = sorted(q2_rows.loc[q2_rows["transect_id"] == transect2, "winter"].unique())
        winter2 = st.selectbox("Winter", winters_here, key="q2_winter")

    row2 = q2_rows[(q2_rows["transect_id"] == transect2) & (q2_rows["winter"] == winter2)]
    live2 = predict_live(q2_models, q2_features, row2).iloc[0]
    actual2 = row2[TARGET].iloc[0]

    st.write(f"**Actual change:** {actual2:.2f} m")
    st.markdown("**This beach's inputs to the models**")
    st.dataframe(beach_inputs_table(row2, q1_rows.drop_duplicates("transect_id")), use_container_width=True)
    st.dataframe(pd.DataFrame({"Predicted change (m)": live2.round(2),
                               "Error: predicted − actual (m)": (live2 - actual2).round(2)}), use_container_width=True)

    st.subheader("The beach, side view")
    shown2 = {"Baseline (overall average)": "Baseline: overall average",
              "Linear": "Linear: beach + wave", "GAM": "GAM: beach + wave",
              "Random forest": "RF: beach + wave", "GBT": "GBT: beach + wave"}
    st.altair_chart(beach_cross_section(actual2, {label: live2[column] for label, column in shown2.items()},
                                        f"{transect2}, winter {winter2} (a beach the models never saw)"),
                    use_container_width=True)
    st.caption("Dashed white line = shoreline before winter · red line = where it actually was after winter · "
               "each dot = where a model (beach + wave features) predicted it would be.")

    st.subheader("What if this beach were steeper or flatter?")
    real_slope = row2["beach_slope"].iloc[0]
    st.write("Drag the beach slope (or press **Play**) and watch each model's prediction move. Everything else about "
             "this beach stays the same. Beach slope is the feature both models found useful. "
             + ("This beach's slope is missing in the data." if pd.isna(real_slope) else f"Its real slope is {real_slope:.3f}."))

    # Every model's prediction for this beach at every slope from 0.03 to 0.20, run live
    slope_grid = np.round(np.arange(0.03, 0.2001, 0.005), 3)
    slope_rows = pd.concat([row2] * len(slope_grid), ignore_index=True)
    slope_rows["beach_slope"] = slope_grid
    slope_models = ["Linear: beach + wave", "GAM: beach + wave", "RF: beach + wave", "GBT: beach + wave"]
    slope_sweep = predict_live(q2_models, q2_features, slope_rows, slope_models)
    slope_sweep.index = slope_grid
    slope_sweep.columns = [name.split(":")[0] for name in slope_sweep.columns]
    baseline2 = live2["Baseline: overall average"]

    everything2 = list(slope_sweep.values.ravel()) + [actual2, 0, baseline2]
    x_domain2 = [min(everything2) - 6, max(everything2) + 6]          # fixed, so the dots visibly slide

    def predictions_at_slope(slope):
        nearest = slope_sweep.iloc[int(np.abs(slope_sweep.index - slope).argmin())]
        at = {"Baseline (overall average)": baseline2}
        for model_name in slope_sweep.columns:
            at[model_name] = nearest[model_name]
        return at

    start_slope = 0.065 if pd.isna(real_slope) else float(np.clip(round(real_slope / 0.005) * 0.005, 0.03, 0.2))
    slope_col, play2_col = st.columns([4, 1])
    with slope_col:
        chosen_slope = st.slider("Beach slope (higher = steeper)", 0.03, 0.20, start_slope, 0.005, format="%.3f",
                                 key="q2_what_if_slope")
    with play2_col:
        st.write("")
        play2 = st.button("▶ Play (flat → steep)", key="q2_play")

    frame2 = st.empty()
    if play2:
        for slope in slope_grid:
            frame2.altair_chart(beach_cross_section(actual2, predictions_at_slope(slope), f"Beach slope = {slope:.3f}",
                                                    x_domain2), use_container_width=True)
            time.sleep(0.12)
    else:
        frame2.altair_chart(beach_cross_section(actual2, predictions_at_slope(chosen_slope),
                                                f"Beach slope = {chosen_slope:.3f}", x_domain2), use_container_width=True)
    st.caption("Dashed white line = shoreline before winter · red line = where it actually ended up · each dot = where a "
               "model predicts it would end up at the chosen slope. The baseline ignores slope, so it never moves. "
               "Most real beaches have slopes between 0.05 and 0.09.")

    with st.expander("Full curves: each model's prediction across all slopes"):
        st.altair_chart(sweep_chart(slope_sweep, "Beach slope", chosen_slope, real_slope, actual2, baseline2),
                        use_container_width=True)
        st.caption("Gray dashed line = baseline (overall average) · gray vertical line = chosen slope · "
                   "red dot = this beach's real slope and real change. Hover for values.")


# ================================================================ This winter
with tab3:
    st.header("What do our models predict for this winter (2026–27)?")
    takeaway("Our models can produce a number for a strong El Niño winter, but it isn't a forecast: RONI didn't reliably "
             "help in our tests, and this winter may be stronger than any winter the models were trained on.")
    st.write("NOAA's latest RONI value is **+2.0** (Aug–Oct 2026), and its outlook calls for a strong-to-very-strong "
             "El Niño this winter. Choose a RONI value: every Question 1 model predicts shoreline change for every "
             "beach along the coast, live.")
    st.info("These are model outputs, not a forecast. Our results showed RONI doesn't reliably improve predictions, "
            f"and the models never saw a winter stronger than RONI {MAX_TRAINING_RONI:+.2f} (2015–16).")

    forecast_a, forecast_b = st.columns(2)
    with forecast_a:
        forecast_roni = st.slider("RONI for this winter", -2.0, 3.0, 2.0, 0.1, key="forecast_roni")
    with forecast_b:
        forecast_model = st.selectbox("Model", RONI_MODELS, index=RONI_MODELS.index("GBT: beach + wave + RONI"),
                                      key="forecast_model")
    if forecast_roni > MAX_TRAINING_RONI:
        st.warning(f"RONI {forecast_roni:+.1f} is outside the training data (max {MAX_TRAINING_RONI:+.2f}). "
                   "Tree models (RF, GBT) treat anything above that like 2015–16; Linear extends its straight line.")

    # One row per beach (beach and wave features are the same every winter), with RONI set to the chosen value
    coast = q1_rows.drop_duplicates("transect_id").copy()
    coast["RONI"] = forecast_roni
    start = time.perf_counter()
    coast["predicted"] = q1_models[forecast_model].predict(coast[q1_features[forecast_model]])
    neutral = coast.assign(RONI=0.0)
    neutral_average = q1_models[forecast_model].predict(neutral[q1_features[forecast_model]]).mean()
    baseline_average = q1_models["Baseline: transect average"].predict(coast[["transect_id"]]).mean()
    seconds = time.perf_counter() - start

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Average predicted change ({forecast_model.split(':')[0]})", f"{coast['predicted'].mean():+.1f} m",
              delta=f"{coast['predicted'].mean() - neutral_average:+.1f} m vs a neutral winter (RONI 0)",
              delta_color="inverse")
    m2.metric("Beaches predicted to erode", f"{(coast['predicted'] < 0).mean() * 100:.0f}%")
    m3.metric("Baseline: each beach's usual winter", f"{baseline_average:+.1f} m")
    st.caption(f"Ran {forecast_model} live on {len(coast):,} beaches ({seconds * 1000:.0f} ms).")

    coast = coast.merge(locations, on="transect_id")
    scale = max(5.0, coast["predicted"].abs().quantile(0.95))
    gradient_legend("Erosion (shoreline moves toward land)", ["rgb(255,0,0)", "rgb(255,255,255)", "rgb(0,0,255)"],
                    "Gain (shoreline moves toward ocean)", f"White = little change · strongest color = ±{scale:.0f} m")
    st.pydeck_chart(coast_forecast_map(coast, scale))
    st.caption("Each dot is one beach (transect). Hover a dot for its value. Needs internet for the background map.")

    st.subheader("How the coast-wide prediction changes with RONI")
    st.altair_chart(coast_average_chart(q1_rows.drop_duplicates("transect_id"), q1_models, q1_features, forecast_roni),
                    use_container_width=True)
    st.caption("Each line = one model's average predicted change across all beaches. "
               f"Gray area = RONI above {MAX_TRAINING_RONI:+.2f}, outside the training data. White line = chosen RONI.")
