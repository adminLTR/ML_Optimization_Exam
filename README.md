#  Project Report

**Course:** Machine Learning Task  
**Date:** June 2026  
**Student:** La Torre Romero, Jose Luis

---

## Table of Contents

1. Summary
2. Project Overview
3. Dataset Description
4. Project Structure
5. Methodology
6. Exploratory Data Analysis Findings
7. Model Development and Results
8. Prediction, Interpretation, and Business Recommendations
9. How to Run the Project

---

## 1. Summary

This project develops an end-to-end machine learning pipeline to predict hourly bike rental demand (`cnt`) for the Capital Bikeshare system in Washington, D.C., using historical data from 2011–2012 combined with weather and calendar features.

The workflow includes data cleaning, anomaly handling, feature engineering, correlation-based feature selection, model training, hyperparameter tuning, and model interpretation.

**Key results on the held-out test set (20%):**

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Linear Regression (baseline) | 126.30 | 91.63 | 0.5010 |
| Random Forest (default) | 41.29 | 25.53 | 0.9467 |
| Random Forest (fine-tuned) | 41.29 | 25.53 | 0.9467 |

The fine-tuned Random Forest outperforms the linear baseline, explaining approximately **94.7%** of the variance in hourly rentals with a mean absolute error of about **26 bikes per hour**.

The most influential predictors are cyclical hour features (`hr_sin`, `hr_cos`), temperature, and year (`yr`), confirming strong temporal and weather-driven demand patterns.

---

## 2. Project Overview

### 2.1 Objective

Build a regression model that predicts the total number of hourly bike rentals (`cnt`) based on:

- Temporal information (date, season, year, month, hour, weekday, holiday, working day)
- Weather conditions (temperature, humidity, wind speed, comfort index, weather situation)

### 2.2 Problem Type

- **Supervised learning**
- **Regression** (continuous target: `cnt`)
- **Evaluation metrics:** RMSE, MAE, and R²

### 2.3 Business Context

Bike-sharing systems are highly sensitive to weather, time of day, and seasonality. Accurate demand forecasting supports:

- Fleet allocation and station rebalancing before peak hours
- Proactive deployment during favorable weather
- Capacity planning as user adoption grows year over year

---

## 3. Dataset Description

### 3.1 Source

The dataset is derived from the Capital Bikeshare system (Washington, D.C., USA), aggregated at hourly resolution and enriched with weather data. Background information is documented in `data/about_bike_sharing_dataset.txt`.

- **Period:** 2011–2012
- **Granularity:** Hourly records
- **Original raw size:** 17,379 rows × 19 columns

### 3.2 Variables

| Column | Description |
|---|---|
| `instant` | Record index |
| `dteday` | Date |
| `season` | Season (1–4) |
| `yr` | Year (0 = 2011, 1 = 2012) |
| `mnth` | Month (1–12) |
| `hr` | Hour (0–23) |
| `holiday` | Whether the day is a holiday |
| `weekday` | Day of week |
| `workingday` | Working day indicator |
| `weathersit` | Weather situation category |
| `temp` | Normalized temperature |
| `atemp` | Normalized feeling temperature |
| `hum` | Normalized humidity |
| `pctghum` | Humidity as percentage string |
| `windspeed` | Normalized wind speed |
| `comfindex` | Comfort index |
| `casual` | Casual user count |
| `registered` | Registered user count |
| `cnt` | **Target:** total rentals |
| `simurevenue` | Simulated hourly revenue |

### 3.3 Target Variable

`cnt` represents total hourly rentals and is defined as:

```text
cnt = casual + registered
```

---

## 4. Project Structure

```text

main.ipynb                      # Main notebook (full pipeline)
README.md                       # This report
REPORT.pdf                      # This report in PDF format
requirements.txt
data/
├── bike_sharing_dataset.csv    # Primary dataset
├── about_bike_sharing_dataset.txt
└── or_data.json                # Bonus OR input
```

---

## 5. Methodology

The notebook is organized into four main sections:

| Section | Weight | Description |
|---|---:|---|
| 1. Data Exploration and Preprocessing | 40% | EDA, cleaning, feature engineering |
| 2. Model Development and Evaluation | 40% | Baseline + Random Forest + tuning |
| 3. Prediction and Interpretation | 20% | Feature importance and business insights |
| 4. Stock Optimization (Bonus) | Extra | OR-based bike redistribution |

### 5.1 Data Type Optimization

- `dteday` converted to `datetime`
- `season`, `holiday`, `workingday`, and `weathersit` converted to categorical dtype

### 5.2 Missing Value Treatment

Missing values were found in:

| Column | Missing Count |
|---|---:|
| `temp` | 132 |
| `hum` / `pctghum` | 192 |
| `comfindex` | 132 |

**Strategy:**

1. Apply **linear interpolation** to `temp`, `hum`, and `comfindex` (time-dependent behavior).
2. Rebuild `pctghum` from interpolated `hum`.

### 5.3 Anomaly Detection and Correction

| Issue | Action |
|---|---|
| Negative `windspeed` values | Replaced with absolute values (sensor sign error assumption) |
| `hum = 0` (impossible) | Replaced with `NaN`, then linear interpolation |
| `cnt = 8500` (5 records) | Removed as physically impossible given fleet size |

After removing the 5 outlier rows, the cleaned dataset has **17,374 records**.

### 5.4 Feature Selection (Correlation Analysis)

A correlation heatmap guided removal of redundant or leaky variables:

| Removed | Reason |
|---|---|
| `atemp` | Near-perfect correlation with `temp` (multicollinearity) |
| `pctghum` | Redundant string version of `hum` |
| `casual`, `registered` | Direct components of target (`cnt`) — data leakage |
| `simurevenue` | Highly correlated with `cnt` — data leakage |

### 5.5 Feature Engineering

#### Cyclical encoding

To preserve periodic structure, sine/cosine transforms were applied to:

- `hr` → `hr_sin`, `hr_cos` (24-hour cycle)
- `mnth` → `mnth_sin`, `mnth_cos` (12-month cycle)
- `weekday` → `weekday_sin`, `weekday_cos` (7-day cycle)

Original raw columns (`hr`, `mnth`, `weekday`) were dropped after encoding.

#### Categorical encoding

One-hot encoding was applied to:

- `season`
- `holiday`
- `workingday`
- `weathersit`

### 5.6 Final Feature Set for Modeling

After preprocessing and encoding, the model uses **19 numeric features** (excluding `dteday` and target `cnt`).

Train/test split:

- **Training:** 13,899 samples (80%)
- **Testing:** 3,475 samples (20%)
- **Random state:** 42

---

## 6. Exploratory Data Analysis Findings

### 6.1 Distribution of Target (`cnt`)

- Right-skewed distribution
- Most hourly counts cluster around 100–200 rentals
- Few very high-demand hours

### 6.2 Temporal Patterns

- **Year-over-year growth:** 2012 rentals are significantly higher than 2011
- **Seasonality:** Peaks in warmer months, lower demand in colder months
- **Weekday vs weekend:** Weekdays show higher average demand and lower variability

### 6.3 Weather Relationships

- **Temperature / comfort index:** Strong positive relationship with rentals
- **Humidity:** Mild negative effect at very high humidity
- **Wind speed:** Very high wind associated with lower demand; calm conditions favor riding

---

## 7. Model Development and Results

### 7.1 Models Evaluated

#### Baseline: Linear Regression

A simple linear model was trained to establish a lower-bound benchmark.

| Metric | Value |
|---|---:|
| RMSE | 126.30 |
| MAE | 91.63 |
| R² | 0.5010 |

The baseline captures only about half of the variance, indicating strong non-linear interactions (especially hour-of-day effects).

#### Advanced Model: Random Forest Regressor

Default configuration with `random_state=42`.

| Metric | Value |
|---|---:|
| RMSE | 41.29 |
| MAE | 25.53 |
| R² | 0.9467 |

This represents a **~67% reduction in RMSE** versus the baseline.

### 7.2 Hyperparameter Tuning (Grid Search)

Grid search was performed with 3-fold cross-validation, optimizing R².

**Search space:**

```python
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
```

**Best parameters found:**

```python
{
    'max_depth': None,
    'min_samples_split': 2,
    'n_estimators': 100
}
```

**Tuned model performance (test set):**

| Metric | Value |
|---|---:|
| RMSE | 41.29 |
| MAE | 25.53 |
| R² | 0.9467 |

Tuning did not materially change performance versus the default Random Forest in this case, suggesting the default configuration was already near-optimal for this dataset.

### 7.3 Model Comparison Summary

| Model | RMSE ↓ | MAE ↓ | R² ↑ |
|---|---:|---:|---:|
| Linear Regression | 126.30 | 91.63 | 0.5010 |
| Random Forest | 41.29 | 25.53 | 0.9467 |
| Random Forest (tuned) | 41.29 | 25.53 | 0.9467 |

**Selected final model:** Fine-tuned Random Forest Regressor (`n_estimators=100`, `max_depth=None`, `min_samples_split=2`).

---

## 8. Prediction, Interpretation, and Business Recommendations

### 8.1 Feature Importance

The Random Forest feature importance plot highlights the main demand drivers:

1. **Cyclical hour features (`hr_sin`, `hr_cos`)** — strongest predictors; demand is highly time-of-day dependent
2. **Temperature (`temp`)** — major weather driver
3. **Year (`yr`)** — captures system growth between 2011 and 2012
4. Additional contributions from month/weekday encodings, comfort index, humidity, wind, season, and weather

### 8.2 Interpretation

Model behavior is consistent with EDA:

- Peak-hour structure dominates predictions
- Warmer, more comfortable conditions increase expected rentals
- Long-term growth trend is embedded in the year feature

### 8.3 Business Recommendations

* Since the hour of the day is the most critical factor, the logistics team must ensure that every station is properly balanced right before peak hours to avoid empty or full situations.
* Temperature significantly drives demand. On unusually warm days, the system should proactively distribute a larger fleet of bikes across the network.
* The strong importance of the `yr` feature highlights a growing user base. The operation should anticipate higher overall baseline demand in the coming years and strongly consider expanding the number of bikes and stations to accommodate this continuous growth.

---

## 9. How to Run the Project

### 9.1 Prerequisites

**Python packages:**

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```

### 9.2 Run in Google Colab

1. Upload the project folder (or upload `main.ipynb` together with the `data/` directory).
2. Open `main.ipynb` in Google Colab.
3. Ensure the notebook can access the dataset at:

   ```text
   data/bike_sharing_dataset.csv
   ```

4. Run all cells sequentially (`Runtime` → `Run all`).
5. Colab already includes most scientific Python libraries. 


### 9.3 Run Locally

#### Step 1 — Clone or copy the project

```bash
git clone https://github.com/adminLTR/ML_Optimization_Exam.git
```

#### Step 2 — Create and activate a virtual environment (recommended)

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

#### Step 4 — Launch Jupyter and open the notebook

```bash
pip install notebook
jupyter notebook main.ipynb
```

Or with JupyterLab:

```bash
pip install jupyterlab
jupyter lab main.ipynb
```

#### Step 5 — Execute the notebook

Run cells from top to bottom. The first data-loading cell expects a relative path:

```python
df = pd.read_csv('data/bike_sharing_dataset.csv')
```

Therefore, the notebook kernel must start in the project root.