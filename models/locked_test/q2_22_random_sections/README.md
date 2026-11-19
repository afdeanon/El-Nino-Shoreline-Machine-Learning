# Question 2 models: locked test set = 22 random coastal sections

Made by `scripts/shoreline_model_comparison_holdout.ipynb`.

- **Locked test sections (never used for training or tuning):** 22 of the 110 coastal sections, picked at random
  with a fixed seed (42). The list is in `model_info.json` under `"locked sections"`.
- **Training data:** the other 88 sections, all winters.
- Predicting `test_rows.csv` with these models reproduces the "Test" scores for test set A in `../q2_results.csv`.
- Saved with **scikit-learn 1.9.0** — load them with the same version.
- No RONI here: Question 2 asks whether beach and wave features explain differences between places.

## Files

| File | What it is |
|---|---|
| `*.pkl` (9 files) | Trained models: baseline (overall average), Linear / GAM / RF / GBT × (beach, beach + wave) |
| `model_info.json` | Which feature columns each `.pkl` expects (in order) |
| `test_rows.csv` | The locked test rows (22 sections, all winters) with all features and the actual `shoreline_change_m` |
| `test_predictions.csv` | Each model's prediction for every test row (one column per model) + `shoreline_change_m` (actual) |

## Loading a model

```python
import json, joblib, pandas as pd

folder = "models/locked_test/q2_22_random_sections"
info  = json.load(open(f"{folder}/model_info.json"))
rows  = pd.read_csv(f"{folder}/test_rows.csv")

model = joblib.load(f"{folder}/gbt_beach_wave.pkl")
predictions = model.predict(rows[info["gbt_beach_wave.pkl"]["features"]])
```

All 9 models load normally (no custom classes needed).

If you only need predictions (not live what-if), use `test_predictions.csv` — no loading needed.
