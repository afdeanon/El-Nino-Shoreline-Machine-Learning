# `models/` — saved models and predictions

Everything in this folder comes from `scripts/shoreline_model_comparison_simplified.ipynb`
(the save cells at the end of that notebook). **You don't need to rerun the notebook** to use these files —
just `git pull`.

> ⚠️ These files were built from the modeling dataset **before** the Nov 18 update
> (new RONI window + revised shoreline target). If the dataset is rebuilt, the notebook and its save
> cells need to be rerun so these files match.

---

## Files at a glance

| File | What it is | Size | Use it for |
|---|---|---|---|
| `heldout_predictions.parquet` | **Predictions** from every model, for every transect × winter | ~2.9 MB | Charts, tables, model comparisons |
| `gbt_beach_wave.pkl` | Trained **GBT**, beach + wave features (no RONI) | ~1.1 MB | Live "what-if" predictions |
| `gbt_beach_wave_roni.pkl` | Trained **GBT**, beach + wave + RONI | ~1.1 MB | Live "what-if" predictions (El Niño slider) |
| `linear_beach_wave.pkl`, `linear_beach_wave_roni.pkl` | Trained **Linear regression** (no RONI / with RONI) | ~1 KB each | Live predictions |
| `gam_beach_wave.pkl`, `gam_beach_wave_roni.pkl` | Trained **GAM** (splines + ridge) | ~2 KB each | Live predictions |
| `rf_beach_wave.pkl`, `rf_beach_wave_roni.pkl` | Trained **Random Forest** (compressed) | ~3.2 MB each | Live predictions |
| `baseline_transect_averages.csv` | **Baseline model**: average shoreline change for each transect | ~117 KB | Baseline prediction = look up the transect's average |
| `model_info.json` | Which features each `.pkl` expects (in order), GBT settings, scikit-learn version | ~2 KB | Read before calling `.predict()` |

**Target** (what every model predicts): `shoreline_change_m` = shoreline position in Mar–Apr minus Oct–Nov, in meters
(negative = erosion).

---

## 1. `heldout_predictions.parquet` — the predictions

**Held-out** predictions: each value was made by a model that did *not* see that row during training.
These are the numbers behind the R² / MAE scores in the notebook.

- One row per **transect × winter** (~60,600 rows)
- Info columns: `transect_id`, `section`, `winter`, `RONI`, `actual_m`
- Then **one column per model + test** (31 columns), named `"<model> | <test>"`, e.g.
  - `Baseline: transect average | test1`
  - `GBT: beach + wave + RONI | test1`
  - `RF: beach + wave | test2`

**Tests**
- `test1` = **new winter** — leave one winter out (21 folds). Can the model predict a winter it hasn't seen?
- `test2` = **new coastline** — hold out coastal sections (10 folds). Can it predict a place it hasn't seen?
- The three `Transect average + …` models only have `test1` (by design).

**Models included:** Baseline (overall average, transect average); Linear, GAM, RF, GBT × feature sets
(`beach`, `beach + wave`, `beach + wave + RONI`); Transect average + linear RONI; Transect average + GBT
(with and without RONI).

```python
import pandas as pd
preds = pd.read_parquet("models/heldout_predictions.parquet")   # needs pyarrow (Streamlit already has it)

# error for one model
col = "GBT: beach + wave + RONI | test1"
preds["error_m"] = preds[col] - preds["actual_m"]

# long format (one row per model/test), if easier for plotting
long = preds.melt(id_vars=["transect_id", "section", "winter", "RONI", "actual_m"],
                  var_name="model_test", value_name="predicted_m")
```

> Parquet is a binary file — double-clicking it in Jupyter gives a "not UTF-8 encoded" error. That's normal;
> open it with `pd.read_parquet`.

---

## 2. `.pkl` files — the trained models

Each `.pkl` is a trained scikit-learn model trained on **all** the data. Load it and call `.predict()`;
no retraining needed.

```python
import joblib, json, pandas as pd

info  = json.load(open("models/model_info.json"))
model = joblib.load("models/gbt_beach_wave_roni.pkl")
features = info["gbt_beach_wave_roni.pkl"]["features"]

# X must have exactly these columns, in this order
X = pd.DataFrame([{
    "BeachWidth_m": 40, "beach_slope": 0.08,
    "shore_orientation_sin": 0.5, "shore_orientation_cos": 0.87,
    "winter_wave_power": 15000, "wave_dir_sin": 0.3, "wave_dir_cos": 0.95,
    "RONI": 2.0,          # strong El Niño
}])[features]
print(model.predict(X))   # predicted shoreline change (m)
```
*(The numbers above are placeholders — use real feature values from the modeling dataset.)*

**Features**
- No-RONI models: `BeachWidth_m, beach_slope, shore_orientation_sin, shore_orientation_cos, winter_wave_power, wave_dir_sin, wave_dir_cos`
- RONI models: the same seven **+ `RONI`**

**GBT settings** (picked by grid search, tuned with winters kept together):
`max_leaf_nodes=31, min_samples_leaf=100, learning_rate=0.05, max_iter=300` with early stopping.

**Linear / GAM / RF** are pipelines that fill missing values themselves (median imputer), so NaNs in
`beach_slope` are OK. GBT handles NaNs natively.

> ⚠️ Load `.pkl` files with the **same scikit-learn version** that saved them (**1.9.0**, see
> `model_info.json`). A different version may fail to load or warn.

---

## 3. `baseline_transect_averages.csv` — the baseline

Two columns: `transect_id`, `avg_change_m`. The baseline "prediction" for a transect is just its average
change across all winters.

```python
baseline = pd.read_csv("models/baseline_transect_averages.csv", index_col="transect_id")["avg_change_m"]
baseline["<some transect_id>"]
```

In Test 1 (new winter) this simple baseline scored as well as or better than every other model — worth showing
next to the GBT predictions in the demo.

---

## Demo tips
- Use a transect that's **in the dataset**. Models do well on a new winter at a known beach (Test 1) but
  close to zero on unseen coastline (Test 2).
- Nice demo: same beach, slide RONI from −2 (La Niña) to +2 (El Niño) using the RONI models and compare
  with the baseline — the prediction barely moves, which is the project's main finding.
