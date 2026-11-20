# Question 1 models: locked test set = last 4 winters

Made by `scripts/shoreline_model_comparison_holdout.ipynb`.

- **Locked test winters (never used for training or tuning):** 2017–18, 2018–19, 2019–20, 2020–21
- **Training data:** the other 17 winters (2000–01 through 2016–17)
- Predicting `test_rows.csv` with these models reproduces the "Test" scores for test set B in `../q1_results.csv`.
- Saved with **scikit-learn 1.9.0** — load them with the same version.

## Files

| File | What it is |
|---|---|
| `*.pkl` (17 files) | Trained models: baselines, Linear / GAM / RF / GBT × (beach, beach + wave, beach + wave + RONI), and 3 "transect average + …" models |
| `model_info.json` | Which feature columns each `.pkl` expects (in order) |
| `test_rows.csv` | The locked test rows (4 winters) with all features and the actual `shoreline_change_m` |
| `test_predictions.csv` | Each model's prediction for every test row (one column per model) + `shoreline_change_m` (actual) |

## Loading a model

```python
import json, joblib, pandas as pd

folder = "models/locked_test/q1_last_4_winters"
info  = json.load(open(f"{folder}/model_info.json"))
rows  = pd.read_csv(f"{folder}/test_rows.csv")

model = joblib.load(f"{folder}/gbt_beach_wave_roni.pkl")
predictions = model.predict(rows[info["gbt_beach_wave_roni.pkl"]["features"]])
```

## Important: 4 files need extra code to load

`baseline_transect_average.pkl` and the three `transect_average_*.pkl` files use custom classes
(`TransectAverage`, `TransectAveragePlus`) defined in the notebook. To load them, paste the two class
definitions from Section 3 of `shoreline_model_comparison_holdout.ipynb` into your script **before** calling
`joblib.load`. In Streamlit, also add these lines after the class definitions (pickle looks for the classes in `__main__`):

```python
import __main__
__main__.TransectAverage = TransectAverage
__main__.TransectAveragePlus = TransectAveragePlus
```

These 4 files also need **pyarrow** installed (pandas stored the transect names in pyarrow format).
Streamlit installs pyarrow automatically. If you only see these 4 fail, run `pip install pyarrow`.

All other `.pkl` files load normally.

If you only need predictions (not live what-if), use `test_predictions.csv` — no loading needed.
