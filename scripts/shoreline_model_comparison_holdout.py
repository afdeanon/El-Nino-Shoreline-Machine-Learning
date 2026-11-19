#!/usr/bin/env python
# Model comparison with a locked test set (converted from shoreline_model_comparison_holdout.ipynb).
# Run from the scripts/ folder:  python shoreline_model_comparison_holdout.py   (slow: about 1-2 hours)

def display(x):
    print(x)

# %% [markdown]
# # El Niño Shoreline Project: Model Comparison with a Locked Test Set
#
# This notebook repeats the model comparison from `shoreline_model_comparison_simplified.ipynb`,
# with one change: **about 20% of the data is locked away before anything else happens.**
#
# For each test set:
# 1. **Lock away** the test data (the models never see it until step 4).
# 2. **Cross-validate on the other 80%** with the same method as the main notebook
#    (leave-one-winter-out for Question 1, 10 folds grouped by coastal section for Question 2).
# 3. **Train** each model on all of the 80%.
# 4. **Test once** on the locked data.
#
# To check that the conclusions don't depend on *which* data we lock away, we try several test sets:
#
# | | Test sets tried |
# |---|---|
# | **Question 1** (new winter) | A. 4 winters spread across La Niña → El Niño · B. the last 4 winters · C. 4 random winters |
# | **Question 2** (new coastline) | A. 22 random coastal sections · B. every 5th section along the coast |
#
# The models, features, settings and GBT tuning are the same as in the main notebook.
# Drop-column feature importance (8c) is not repeated here.
#
# **This notebook is slow (likely 1–2 hours).** Each step prints its progress, so you can leave it running.
# Everything this notebook saves goes to `models/locked_test/`. The main notebook and the other files in `models/` are not touched.

# %% [markdown]
# ## 1. Setup (same settings as the main notebook)

# %%
import json
import time
from pathlib import Path

import joblib
import sklearn

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

ROOT_DIR  = Path.cwd().parent           # this notebook lives in scripts/
DATA_PATH = ROOT_DIR / "processed" / "final_modeling_dataset.csv"
OUT_DIR   = ROOT_DIR / "models" / "locked_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "shoreline_change_m"           # Mar–Apr average position minus Oct–Nov average position
OUTLIER_LIMIT_M = 50                    # drop rows where |change| is above this

BEACH_FEATURES = ["BeachWidth_m", "beach_slope", "shore_orientation_sin", "shore_orientation_cos"]
WAVE_FEATURES  = ["winter_wave_power", "wave_dir_sin", "wave_dir_cos"]

FEATURE_SETS = {
    "beach":               BEACH_FEATURES,
    "beach + wave":        BEACH_FEATURES + WAVE_FEATURES,
    "beach + wave + RONI": BEACH_FEATURES + WAVE_FEATURES + ["RONI"],
}

# %% [markdown]
# ## 2. Load the data and remove outliers (same as the main notebook)

# %%
df = pd.read_csv(DATA_PATH)
df = df[df[TARGET].abs() <= OUTLIER_LIMIT_M]
df = df.reset_index(drop=True)
df["section"] = df["transect_id"].str[:11]      # first 11 characters = coastal section

print("Rows:", len(df))
print("Winters:", df["winter"].nunique(), "| Coastal sections:", df["section"].nunique())

# %% [markdown]
# ## 3. The models (copied from the main notebook)

# %%
class TransectAverage(BaseEstimator, RegressorMixin):
    """Baseline: predict each transect's average change in the training data."""
    def fit(self, X, y):
        training = pd.DataFrame({"transect_id": X["transect_id"].values, "change": np.asarray(y)})
        self.averages_ = training.groupby("transect_id")["change"].mean()
        self.overall_ = training["change"].mean()      # backup for transects never seen
        return self

    def predict(self, X):
        predictions = X["transect_id"].map(self.averages_)
        predictions = predictions.fillna(self.overall_)
        return predictions.to_numpy()


class TransectAveragePlus(BaseEstimator, RegressorMixin):
    """Transect average + a second model fit to each row's difference from its transect's average."""
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


def make_linear():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())

def make_gam():
    return make_pipeline(SimpleImputer(strategy="median"), SplineTransformer(knots="quantile"), Ridge())

def make_rf():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=100, min_samples_leaf=50, max_features=0.5, n_jobs=-1, random_state=42),
    )

GBT_GRID = {"max_leaf_nodes": [15, 31], "min_samples_leaf": [100, 500]}

def make_gbt():
    base = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, early_stopping=True,
                                         validation_fraction=0.1, n_iter_no_change=20, random_state=42)
    return GridSearchCV(base, GBT_GRID, cv=GroupKFold(n_splits=3), scoring="neg_mean_absolute_error")

small_gbt = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                          min_samples_leaf=500, random_state=42)

# %% [markdown]
# ### The list of models for each question
#
# Each entry is: name, model, features, and whether the model needs group labels (only the GBT, for its tuning).
# Question 1 uses the same models as Test 1 in the main notebook. Question 2 uses the models from the main notebook's
# Question 2 table (beach and beach + wave; RONI can't tell places apart).

# %%
def question_1_models():
    models = [
        ["Baseline: overall average",  DummyRegressor(strategy="mean"), ["RONI"],        False],
        ["Baseline: transect average", TransectAverage(),               ["transect_id"], False],
    ]
    for set_name, features in FEATURE_SETS.items():
        models.append([f"Linear: {set_name}", make_linear(), features, False])
        models.append([f"GAM: {set_name}",    make_gam(),    features, False])
        models.append([f"RF: {set_name}",     make_rf(),     features, False])
        models.append([f"GBT: {set_name}",    make_gbt(),    features, True])
    models.append(["Transect average + linear RONI",
                   TransectAveragePlus(LinearRegression(), ["RONI"]),
                   ["transect_id", "RONI"], False])
    models.append(["Transect average + GBT beach + wave (control)",
                   TransectAveragePlus(small_gbt, FEATURE_SETS["beach + wave"]),
                   ["transect_id"] + FEATURE_SETS["beach + wave"], False])
    models.append(["Transect average + GBT beach + wave + RONI",
                   TransectAveragePlus(small_gbt, FEATURE_SETS["beach + wave + RONI"]),
                   ["transect_id"] + FEATURE_SETS["beach + wave + RONI"], False])
    return models


def question_2_models():
    models = [["Baseline: overall average", DummyRegressor(strategy="mean"), ["RONI"], False]]
    for set_name in ["beach", "beach + wave"]:
        features = FEATURE_SETS[set_name]
        models.append([f"Linear: {set_name}", make_linear(), features, False])
        models.append([f"GAM: {set_name}",    make_gam(),    features, False])
        models.append([f"RF: {set_name}",     make_rf(),     features, False])
        models.append([f"GBT: {set_name}",    make_gbt(),    features, True])
    return models

# %% [markdown]
# ## 4. The four steps, for one test set
#
# `run_test_set` does steps 1–4 for every model and returns a table with two scores per model:
# - **CV** = cross-validation on the 80% (used to compare models, like the main notebook)
# - **Test** = one final check on the locked 20%
#
# If `save_folder` is given, it also saves (for the live demo):
# - every final model (trained on the 80%) as a `.pkl` file
# - `test_rows.csv`: the locked test rows, with all features
# - `test_predictions.csv`: each model's predictions for the locked test rows
# - `model_info.json`: which features each model uses, and which data was locked away

# %%
def file_name(model_name):
    """Turn a model name into a file name, e.g. "GBT: beach + wave" -> "gbt_beach_wave.pkl"."""
    name = model_name.lower()
    for old, new in [(" + ", "_"), (": ", "_"), (" ", "_"), ("(", ""), (")", "")]:
        name = name.replace(old, new)
    return name + ".pkl"


def run_test_set(test_name, is_test, group_column, splitter, models, save_folder=None):
    # Step 1: lock away the test data
    dev  = df[~is_test]       # the 80%: used for cross-validation and training
    test = df[is_test]        # the 20%: not touched until step 4
    print(f"\n=== {test_name} ===")
    print(f"Training rows: {len(dev):,} | Locked test rows: {len(test):,}")

    rows = []
    saved_predictions = {}     # used only when save_folder is given
    model_info = {}
    for name, model, features, needs_groups in models:
        start = time.time()
        groups = dev[group_column]

        # Step 2: cross-validation on the 80% only
        if needs_groups:
            extra = {"groups": groups}        # the GBT's tuning keeps groups together too
        else:
            extra = None
        cv_predictions = cross_val_predict(model, dev[features], dev[TARGET],
                                           groups=groups, cv=splitter, params=extra)

        # Step 3: train on all of the 80%
        final_model = clone(model)
        if needs_groups:
            final_model.fit(dev[features], dev[TARGET], groups=groups)
        else:
            final_model.fit(dev[features], dev[TARGET])

        # Step 4: test once on the locked data
        test_predictions = final_model.predict(test[features])

        rows.append({
            "test set": test_name,
            "model": name,
            "CV R²":        r2_score(dev[TARGET], cv_predictions),
            "CV MAE (m)":   mean_absolute_error(dev[TARGET], cv_predictions),
            "Test R²":      r2_score(test[TARGET], test_predictions),
            "Test MAE (m)": mean_absolute_error(test[TARGET], test_predictions),
        })
        print(f"  {name:48s} done in {time.time() - start:5.0f} s")

        # Save the final model and its predictions (only for the test sets we demo)
        if save_folder is not None:
            if needs_groups:
                model_to_save = final_model.best_estimator_   # the GBT: keep only the winning settings
            else:
                model_to_save = final_model
            joblib.dump(model_to_save, save_folder / file_name(name), compress=3)
            saved_predictions[name] = test_predictions
            model_info[file_name(name)] = {"model": name, "features": features}

    if save_folder is not None:
        id_columns = ["transect_id", "section", "winter"]
        all_features = FEATURE_SETS["beach + wave + RONI"]
        test[id_columns + all_features + [TARGET]].to_csv(save_folder / "test_rows.csv", index=False)

        prediction_table = test[id_columns + ["RONI", TARGET]].copy()
        for name, predictions in saved_predictions.items():
            prediction_table[name] = predictions
        prediction_table.to_csv(save_folder / "test_predictions.csv", index=False)

        model_info["locked test set"] = test_name
        model_info["locked " + group_column + "s"] = sorted(test[group_column].unique().tolist())
        model_info["sklearn_version"] = sklearn.__version__
        with open(save_folder / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        print(f"  Saved {len(saved_predictions)} models + test rows + predictions to {save_folder.relative_to(ROOT_DIR)}")

    return pd.DataFrame(rows)

# %% [markdown]
# ---
# # Question 1: Does El Niño (RONI) help predict a new winter?
#
# ## 5. Choose the locked test winters (about 20% = 4 of 21 winters)
#
# - **A. Spread across El Niño / La Niña:** sort the winters by RONI and take 4 evenly spaced ones, including the
#   strongest La Niña and the strongest El Niño.
# - **B. Last 4 winters:** the most recent winters, like predicting the future.
# - **C. 4 random winters:** picked at random with a fixed seed (42).
#
# All three were chosen by rule, before seeing any results.

# %%
winter_roni = df.groupby("winter")["RONI"].first().sort_values()     # winters from La Niña to El Niño
all_winters = sorted(df["winter"].unique())                          # winters in time order

positions = np.linspace(0, len(winter_roni) - 1, 4).round().astype(int)   # 4 evenly spaced positions
rng = np.random.default_rng(42)

Q1_TEST_SETS = {
    "A: spread across El Niño/La Niña": list(winter_roni.index[positions]),
    "B: last 4 winters":                all_winters[-4:],
    "C: 4 random winters":              sorted(rng.choice(all_winters, size=4, replace=False).tolist()),
}

for test_name, winters in Q1_TEST_SETS.items():
    print(test_name)
    for winter in winters:
        print(f"    {winter}  RONI = {winter_roni[winter]:+.2f}")

# %% [markdown]
# ## 6. Run Question 1 (slow: the GBT tuning runs inside every one of the 17 folds)
#
# Cross-validation on the 80% uses **leave-one-winter-out**, the same as Test 1 in the main notebook.
#
# The models for **test set B (last 4 winters)** are saved to `models/locked_test/q1_last_4_winters/` for the live demo.

# %%
q1_tables = []
overall_start = time.time()
Q1_SAVE_FOLDER = OUT_DIR / "q1_last_4_winters"
Q1_SAVE_FOLDER.mkdir(parents=True, exist_ok=True)

for test_name, winters in Q1_TEST_SETS.items():
    is_test = df["winter"].isin(winters)

    # Save the models only for test set B (last 4 winters)
    if test_name.startswith("B"):
        save_folder = Q1_SAVE_FOLDER
    else:
        save_folder = None

    table = run_test_set(test_name, is_test, "winter", LeaveOneGroupOut(), question_1_models(), save_folder)
    q1_tables.append(table)

q1_results = pd.concat(q1_tables, ignore_index=True)
q1_results.to_csv(OUT_DIR / "q1_results.csv", index=False)
print(f"\nQuestion 1 finished in {(time.time() - overall_start) / 60:.0f} minutes. Saved to models/locked_test/q1_results.csv")

# %% [markdown]
# ## 7. Question 1 results
#
# ### 7a. Test R² for every model, under each locked test set
# Higher is better. The bar to beat is **Baseline: transect average**.

# %%
print(q1_results.pivot(index="model", columns="test set", values="Test R²").loc[q1_results["model"].unique()].round(3))

# %% [markdown]
# ### 7b. Does adding RONI help? (change in Test R² from adding RONI)
# Negative = RONI made predictions worse on the locked winters.

# %%
PAIRS = {
    "Transect average":       ("Baseline: transect average", "Transect average + linear RONI"),
    "Transect average + GBT": ("Transect average + GBT beach + wave (control)", "Transect average + GBT beach + wave + RONI"),
    "Linear":                 ("Linear: beach + wave", "Linear: beach + wave + RONI"),
    "GAM":                    ("GAM: beach + wave",    "GAM: beach + wave + RONI"),
    "Random forest":          ("RF: beach + wave",     "RF: beach + wave + RONI"),
    "GBT":                    ("GBT: beach + wave",    "GBT: beach + wave + RONI"),
}

test_r2 = q1_results.pivot(index="model", columns="test set", values="Test R²")
roni_effect = {}
for family, (without_roni, with_roni) in PAIRS.items():
    roni_effect[family] = test_r2.loc[with_roni] - test_r2.loc[without_roni]

print(pd.DataFrame(roni_effect).T.round(3))

# %% [markdown]
# ### 7c. Cross-validation (80%) vs. locked test (20%)
# If the CV scores and the test scores tell the same story, the model choices made with cross-validation hold up on unseen winters.

# %%
print(q1_results.set_index(["test set", "model"]).round(3))

# %% [markdown]
# ---
# # Question 2: Do beach and wave features explain shoreline change on new coastline?
#
# ## 8. Choose the locked test sections (about 20% = 22 of 110 sections)
#
# - **A. 22 random sections:** picked at random with a fixed seed (42).
# - **B. Every 5th section along the coast:** spread evenly from south to north.

# %%
all_sections = sorted(df["section"].unique())
rng = np.random.default_rng(42)

Q2_TEST_SETS = {
    "A: 22 random sections":  sorted(rng.choice(all_sections, size=22, replace=False).tolist()),
    "B: every 5th section":   all_sections[2::5],
}

for test_name, sections in Q2_TEST_SETS.items():
    print(f"{test_name} ({len(sections)} sections): {', '.join(sections)}")

# %% [markdown]
# ## 9. Run Question 2
#
# Cross-validation on the 80% uses **10 folds grouped by coastal section**, the same as Test 2 in the main notebook.
#
# The models for **test set A (22 random sections)** are saved to `models/locked_test/q2_22_random_sections/` for the live demo.

# %%
q2_tables = []
overall_start = time.time()
Q2_SAVE_FOLDER = OUT_DIR / "q2_22_random_sections"
Q2_SAVE_FOLDER.mkdir(parents=True, exist_ok=True)

for test_name, sections in Q2_TEST_SETS.items():
    is_test = df["section"].isin(sections)

    # Save the models only for test set A (22 random sections)
    if test_name.startswith("A"):
        save_folder = Q2_SAVE_FOLDER
    else:
        save_folder = None

    table = run_test_set(test_name, is_test, "section", GroupKFold(n_splits=10), question_2_models(), save_folder)
    q2_tables.append(table)

q2_results = pd.concat(q2_tables, ignore_index=True)
q2_results.to_csv(OUT_DIR / "q2_results.csv", index=False)
print(f"\nQuestion 2 finished in {(time.time() - overall_start) / 60:.0f} minutes. Saved to models/locked_test/q2_results.csv")

# %% [markdown]
# ## 10. Question 2 results
#
# ### 10a. Test R² for every model, under each locked test set
# Higher is better. The bar to beat is **Baseline: overall average** (an unseen beach has no history to average).

# %%
print(q2_results.pivot(index="model", columns="test set", values="Test R²").loc[q2_results["model"].unique()].round(3))

# %% [markdown]
# ### 10b. Cross-validation (80%) vs. locked test (20%)

# %%
print(q2_results.set_index(["test set", "model"]).round(3))

# %% [markdown]
# ---
# ## Notes
#
# - **No data leakage:** each test set is removed before cross-validation, training and tuning.
#   Baselines are computed from the training data only.
# - **Same methodology as the main notebook,** except for the locked test sets: same models, features, settings,
#   outlier rule, leave-one-winter-out (Q1) and 10-fold grouped CV (Q2), with GBT tuning inside each fold.
# - **Several test sets:** a single locked test set can be easier or harder than average.
#   If the conclusions hold across all of them, they don't depend on which data we held out.
# - **Saved models for the demo:** `models/locked_test/q1_last_4_winters/` and `models/locked_test/q2_22_random_sections/`.
#   They were trained on the 80% only, so predicting `test_rows.csv` reproduces the "Test" scores above.
#   The baseline and "transect average + …" models use classes defined in this notebook (`TransectAverage`,
#   `TransectAveragePlus`); to load those `.pkl` files elsewhere, define the same classes first.
# - **Test set A for Question 1 locks away 2015–16** (the strongest El Niño), so the models train without it.
#   That is a realistic check: a new winter can be stronger than anything in the training data.
