from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed"
MODELS = ROOT / "models"
LOCKED = MODELS / "locked_test"

st.set_page_config(
    page_title="El Niño Shoreline Lab",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_dataset():
    df = pd.read_csv(PROCESSED / "final_modeling_dataset.csv")
    df["section"] = df["transect_id"].str[:11]
    return df


@st.cache_data
def load_locked_results(question):
    return pd.read_csv(LOCKED / f"{question}_results.csv")


@st.cache_data
def load_q2_predictions():
    return pd.read_csv(LOCKED / "q2_22_random_sections" / "test_predictions.csv")


@st.cache_data
def load_coordinates():
    coordinates = pd.read_csv(PROCESSED / "master_transect_features.csv")
    return coordinates[
        ["transect_id", "Lat_AvShoreline_WGS84", "Lon_AvShoreline_WGS84"]
    ].drop_duplicates("transect_id")


def score_chart(results, score_col="Test R²", title=""):
    import plotly.express as px

    chart = results.copy()
    chart["short_model"] = chart["model"].str.replace(
        r"^(Baseline: |Linear: |GAM: |RF: |GBT: )", "", regex=True
    )
    chart = chart.sort_values(score_col)
    fig = px.bar(
        chart,
        x=score_col,
        y="short_model",
        color=score_col,
        color_continuous_scale="RdYlGn",
        orientation="h",
        title=title,
        labels={"short_model": "Model", score_col: score_col},
    )
    fig.update_layout(height=max(420, len(chart) * 30), coloraxis_showscale=False)
    return fig


def apply_style():
    st.markdown(
        """
        <style>
        .hero { padding: 1.3rem 1.5rem; border-radius: 18px; background:
          linear-gradient(120deg, #073b4c 0%, #0b5d72 55%, #118ab2 100%);
          color: white; margin-bottom: 1rem; }
        .hero h1 { margin: 0; font-size: 2.5rem; }
        .hero p { margin: .5rem 0 0; color: #d8f3f9; font-size: 1.05rem; }
        .note { padding: .8rem 1rem; border-left: 4px solid #118ab2; background: #eef8fb; border-radius: 5px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_style()
df = load_dataset()

st.markdown(
    """
    <div class="hero">
      <h1>🌊 El Niño Shoreline Lab</h1>
      <p>Explore how beach morphology, winter waves, and ENSO conditions relate to seasonal shoreline change in Southern California.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Navigate")
    page = st.radio(
        "View",
        [
            "1. Overview & scorecard",
            "2. Test 1 · Temporal prediction",
            "3. Test 2 · Spatial prediction",
            "4. Cross-test comparison",
            "5. Feature interpretation",
            "6. Baselines & methodology",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Target: March–April shoreline position minus October–November position. Negative values indicate retreat.")
    st.caption(f"Dataset: {len(df):,} transect × winter rows")


if page == "1. Overview & scorecard":
    st.subheader("Project at a glance")
    st.write(
        "This dashboard presents the shoreline-change modeling project and its spatial generalization results."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transects", f"{df.transect_id.nunique():,}")
    c2.metric("Winters", f"{df.winter.nunique():,}")
    c3.metric("Coastal sections", f"{df.section.nunique():,}")
    c4.metric("Median change", f"{df.shoreline_change_m.median():.1f} m")

    st.markdown("#### Current scorecard · Test 2")
    q2_overview = load_locked_results("q2")
    q2_latest = q2_overview[q2_overview["test set"] == q2_overview["test set"].iloc[0]].copy()
    st.dataframe(
        q2_latest[["model", "Test R²", "Test MAE (m)"]]
        .sort_values("Test R²", ascending=False)
        .head(8)
        .style.format({"Test R²": "{:.3f}", "Test MAE (m)": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### How to read the dashboard")
    st.markdown(
        """
        <div class="note">
        <b>New coastline</b> means entire coastal sections are held out, so the model cannot rely on neighboring transects.<br><br>
        The locked-test results are the most conservative evidence; the Prediction Lab uses saved models trained on the full dataset for demonstration.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Main finding")
    st.info("Beach and wave features provide limited spatial generalization to coastline sections the model has never seen.")

elif page == "2. Test 1 · Temporal prediction":
    st.subheader("Test 1")
    st.info("This section is reserved for your partner's work.")

elif page == "3. Test 2 · Spatial prediction":
    st.subheader("Test 2 · Spatial prediction")
    st.write("This view is intentionally limited to Test 2: performance on coastline sections held out from training.")
    results = load_locked_results("q2")
    sets = results["test set"].unique().tolist()
    selected = st.selectbox("Locked Test 2 set", sets)
    subset = results[results["test set"] == selected].copy()
    st.plotly_chart(score_chart(subset, title="Test 2 locked-test R² · higher is better"), use_container_width=True)
    st.dataframe(
        subset[["model", "CV R²", "CV MAE (m)", "Test R²", "Test MAE (m)"]]
        .sort_values("Test R²", ascending=False)
        .style.format({"CV R²": "{:.3f}", "Test R²": "{:.3f}", "CV MAE (m)": "{:.2f}", "Test MAE (m)": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

    q2_preds = load_q2_predictions()
    actual_col = "shoreline_change_m" if "shoreline_change_m" in q2_preds.columns else "actual_m"
    excluded = {"transect_id", "section", "winter", "RONI", actual_col}
    prediction_cols = [c for c in q2_preds.columns if c not in excluded]

    st.markdown("#### Actual versus predicted")
    import plotly.express as px

    parity_model = st.selectbox("Model", prediction_cols, key="q2_parity_model")
    parity = q2_preds[[actual_col, parity_model]].dropna()
    parity_fig = px.scatter(
        parity,
        x=actual_col,
        y=parity_model,
        opacity=0.35,
        labels={actual_col: "Observed shoreline change (m)", parity_model: "Predicted shoreline change (m)"},
        title=f"Test 2 parity plot · {parity_model}",
    )
    lo = min(parity[actual_col].min(), parity[parity_model].min())
    hi = max(parity[actual_col].max(), parity[parity_model].max())
    parity_fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line_dash="dash", line_color="gray")
    parity_fig.update_layout(height=500)
    st.plotly_chart(parity_fig, use_container_width=True)

    st.markdown("#### Coastline error map")
    map_model = st.selectbox("Map model", prediction_cols, key="q2_map_model")
    map_data = q2_preds[["transect_id", map_model, actual_col]].copy()
    map_data["absolute_error_m"] = (map_data[map_model] - map_data[actual_col]).abs()
    map_data = map_data.groupby("transect_id", as_index=False)["absolute_error_m"].mean()
    map_data["section"] = map_data["transect_id"].str[:11]
    map_data = map_data.merge(load_coordinates(), on="transect_id", how="left").dropna(
        subset=["Lat_AvShoreline_WGS84", "Lon_AvShoreline_WGS84"]
    )
    map_fig = px.scatter(
        map_data,
        x="Lon_AvShoreline_WGS84",
        y="Lat_AvShoreline_WGS84",
        color="absolute_error_m",
        size="absolute_error_m",
        hover_name="transect_id",
        hover_data={"section": True, "absolute_error_m": ":.2f"},
        color_continuous_scale="YlOrRd",
        labels={"Lon_AvShoreline_WGS84": "Longitude", "Lat_AvShoreline_WGS84": "Latitude", "absolute_error_m": "Mean absolute error (m)"},
        title=f"Mean absolute error by held-out transect · {map_model}",
    )
    map_fig.update_yaxes(scaleanchor="x", scaleratio=1)
    map_fig.update_layout(height=550)
    st.plotly_chart(map_fig, use_container_width=True)

elif page == "4. Cross-test comparison":
    st.subheader("Cross-test comparison")
    st.info("This section is reserved until Test 1 and Test 2 are ready to be compared.")

elif page == "5. Feature interpretation":
    st.subheader("Feature interpretation")
    st.info("This section is reserved for your partner's feature-interpretation work.")

else:
    st.subheader("Baselines & methodology")
    st.info("This section is reserved for your partner's baseline and methodology documentation.")

