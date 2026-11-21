# Does El Niño Help Predict Southern California Shoreline Erosion?

**DATA 245 Machine Learning · San José State University · Fall 2026 · Group 2**
Liana Pakingan · Angela Wei · Angela Faith Deanon · Benita Isaac

This project tests whether an El Niño index, the **Relative Oceanic Niño Index (RONI)**, improves machine learning
predictions of how much Southern California beaches erode each winter, once each beach's own characteristics are known.

We built a dataset of **60,864 transect-winters** (3,220 beach transects × 21 winters, 2000–01 to 2020–21) from
satellite-derived shorelines, then compared linear regression, a generalized additive model (GAM), a random forest,
and a gradient-boosted regression tree (GBT), each with and without RONI, against two simple baselines.

---

## Research questions

1. **New winter:** Does adding RONI improve predictions of winter shoreline change for a winter the model has never seen?
2. **New coastline:** Do beach and wave characteristics explain shoreline change on stretches of coast the model has never seen?

## Key findings

| | Result |
|---|---|
| **Best predictor** | Each beach's own average change from past winters (test R² = 0.45, MAE = 6.07 m). No model meaningfully beat it. |
| **Best ML model** | Gradient-boosted tree with beach + wave features (test R² = 0.39), mostly by recognizing individual beaches. |
| **Effect of RONI** | Adding RONI made every model worse in cross-validation. Flexible tree models used it to memorize individual winters (overfitting). |
| **New coastline** | Every model scored close to zero (R² ≈ 0) on unseen coastal sections. Beach slope was the most useful feature, but only slightly. |

**Bottom line:** a single winter-average El Niño number does not reliably improve local, winter-scale erosion
predictions. A beach's own history is a better guide.

---

## Data

| Source | What we used | Link |
|---|---|---|
| CoastSat Pacific Rim shorelines (Vos) | Tidally corrected shoreline time series and transect beach slopes | [Zenodo](https://doi.org/10.5281/zenodo.7758183) |
| USGS California shoreline seasonality data (Warrick & Buscombe, 2025) | Beach width, shoreline orientation, transect coordinates | [doi:10.5066/P14WWHOJ](https://doi.org/10.5066/P14WWHOJ) |
| Coastal Data Information Program (CDIP), Scripps | Monthly average wave power and direction for each transect | [cdip.ucsd.edu](https://cdip.ucsd.edu) |
| NOAA Climate Prediction Center | Relative Oceanic Niño Index (RONI) | [RONI page](https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/) |

> **Note:** `raw_data/` (~2 GB) and `processed/` (~190 MB) are not in this repository (see `.gitignore`).
> To rebuild them, download the sources above into `raw_data/` and follow [Reproducing the results](#reproducing-the-results).
> You do **not** need them to run the demo or to inspect the saved models and results.

### Target variable

```
shoreline_change_m = average shoreline position in Mar–Apr − average position in the preceding Oct–Nov
```

Negative values mean erosion. Rows with more than 50 m of change in one winter (280 rows, 0.46%) are treated as
satellite errors and removed before modeling.

### Features

| Group | Features |
|---|---|
| Beach | beach width, beach slope, shoreline orientation (sin, cos) |
| Wave | winter (Oct–Apr) average wave power, wave direction (sin, cos) |
| El Niño | RONI, averaged over the five overlapping seasons from OND to FMA |

Only RONI changes from one winter to the next. Beach and wave features are the same every year at a given transect.

---

## Evaluation design

Every model goes through the same four steps:

1. **Lock away** a test set before anything else.
2. **Cross-validate** on the training data to compare models.
3. **Retrain** each model on all of the training data.
4. **Test once** on the locked test set.

| | Question 1: new winter | Question 2: new coastline |
|---|---|---|
| Locked test set | Last 4 winters (2017–18 to 2020–21), 12,495 rows | 22 random coastal sections (seed 42), 16,657 rows |
| Training set | Other 17 winters | Other 88 sections |
| Cross-validation | Leave one winter out (17 folds) | 10 folds grouped by coastal section |
| Feature sets | beach · beach + wave · beach + wave + RONI | beach · beach + wave |

The GBT is tuned with a nested grid search (inner 3-fold split grouped the same way), so the data being predicted
never influences its settings. The conclusions were checked on 3 different sets of test winters and 2 different sets
of test sections (`shoreline_model_comparison_holdout.ipynb`).

Metrics: R², MAE, RMSE, median absolute error, explained variance and maximum error, plus a comparison of training,
cross-validation and test R² to check for overfitting.

---

## Repository structure

```
├── scripts/
│   ├── build_transects.py                                 # California transect lookup table → raw_data/
│   ├── build_roni.py                                      # Downloads RONI and builds winter averages → raw_data/RONI.csv
│   ├── build_modeling_dataset.ipynb                       # Builds processed/final_modeling_dataset.csv, plus EDA
│   ├── shoreline_model_comparison_holdout_simplified.ipynb  # FINAL models: locked test sets (Q1 and Q2)
│   ├── shoreline_model_comparison_holdout.ipynb           # Robustness check across several test sets
│   ├── shoreline_model_comparison_simplified.ipynb        # Earlier version: cross-validation only
│   ├── report_figures_and_metrics.ipynb                   # Report figures and the extra metrics (no training)
│   └── streamlit_demo_app.py                              # Interactive demo using the saved locked-test models
├── models/
│   ├── locked_test/                                       # FINAL saved models, test rows, predictions and results
│   │   ├── q1_last_4_winters/                             # 17 models for Question 1
│   │   ├── q2_22_random_sections/                         # 9 models for Question 2
│   │   ├── q1_results.csv, q2_results.csv                 # Train, CV and test scores for every model
│   │   └── q2_feature_importance.csv                      # Drop-column feature importance (Question 2)
│   └── *.pkl, heldout_predictions.parquet                 # Earlier version (from shoreline_model_comparison_simplified.ipynb)
├── streamlit_app.py                                       # Second dashboard (needs processed/ data locally)
├── raw_data/                                              # Not in the repo: source data
└── processed/                                             # Not in the repo: built datasets
```

### Saved models

The **final** models used in the report and the demo are in **`models/locked_test/`**. Each folder has:

- one `.pkl` file per trained model (saved with `joblib`, scikit-learn 1.9.0),
- `model_info.json` listing the feature columns each model expects, in order,
- `test_rows.csv` (the locked test rows with all features) and `test_predictions.csv` (every model's predictions),
- a `README.md` explaining how to load the models.

The `.pkl` files directly under `models/` come from the earlier cross-validation-only version of the project and are
kept for reference.

---

## Running the demo

The demo loads the saved models and predicts the locked test data live. Nothing is retrained, so it runs in seconds,
and it works from this repository alone (no `raw_data/` or `processed/` needed).

```bash
pip install pandas numpy scikit-learn==1.9.0 joblib streamlit altair pydeck
streamlit run scripts/streamlit_demo_app.py
```

Use scikit-learn 1.9.0 so the saved models load correctly. The app checks that its live scores match the notebook's
saved results.

---

## Reproducing the results

1. **Set up Python** (3.9 or newer) and install the packages:
   ```bash
   pip install pandas numpy scikit-learn==1.9.0 joblib matplotlib pyarrow geopandas contextily streamlit altair pydeck
   ```
2. **Download the source data** listed above into `raw_data/`
   (CoastSat shorelines into `raw_data/shoreline_data/shoreline_data_run6/USA_CA/`).
3. **Build the lookup tables:**
   ```bash
   python scripts/build_transects.py
   python scripts/build_roni.py
   ```
4. **Build the modeling dataset:** run `scripts/build_modeling_dataset.ipynb`
   (creates `processed/final_modeling_dataset.csv`: 60,864 rows, 3,220 transects, 21 winters).
5. **Train and evaluate the models:** run `scripts/shoreline_model_comparison_holdout_simplified.ipynb`
   (about 25–45 minutes, mostly the GBT's nested tuning). This saves everything in `models/locked_test/`.
6. **Optional robustness check:** run `scripts/shoreline_model_comparison_holdout.ipynb`.
7. **Report figures and extra metrics:** run `scripts/report_figures_and_metrics.ipynb` (under a minute).

---

## Contributions

| Team member | Contributions |
|---|---|
| Benita Isaac | Generalized additive model, gradient-boosted regression tree, performance analysis, report |
| Liana Pakingan | Data curation, modeling dataset and preprocessing pipeline |
| Angela Faith Deanon | Linear regression, Question 2 visualizations |
| Angela Wei | Linear regression, Question 1 visualizations |

Parts of this README, the report text and `report_figures_and_metrics.ipynb` were drafted with help from Claude
(Anthropic) and reviewed by the team.
