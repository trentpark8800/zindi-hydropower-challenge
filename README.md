# IBM SkillsBuild Hydropower Climate Optimisation Challenge Solution Overview

# Original Setup

* Python version 3.10.0 was used for the notebook interpreter
* Packages used: pandas, numpy, seaborn, scikit-learn, darts, matplotlib, optuna. For version locking the required packages can be found in `requirements.txt` and installed via `$ pip install -r ./requirements.txt`
* Assumed setup is the the `final_submission.ipynb` is at the root level, while uncompressed data is contained in a folder called `./data/`

# Solution Overview

## Overview and Objectives

This notebook presents the final solution for the IBM SkillsBuild Hydropower Climate Optimisation Challenge. The task is to predict future energy load generation (in kWh) in the remote Kalam region of Pakistan using a combination of micro-hydropower plant (MHP) data and local climate data.

### Purpose
The aim of this solution is to enable accurate 31-day-ahead forecasting of daily power usage, which is crucial for optimizing energy distribution, reducing waste, and improving the resilience of off-grid communities reliant on hydropower.

### Objectives
- Efficiently handle and transform large-scale hydropower and climate datasets using DuckDB to ensure scalability and memory efficiency.
- Aggregate raw data into a consistent daily granularity, aligning all time series for effective modeling.
- Engineer relevant features from both power and climate data to enhance predictive capability.
- Develop and evaluate a robust forecasting model using the Darts `LinearRegressionModel`, leveraging lagged features and covariates.
- Document the entire ETL and modeling pipeline for transparency, reproducibility, and deployment readiness.

This solution focuses on simplicity, efficiency, and interpretability while maintaining strong forecasting performance.

## Architecture Diagram

![image.png](attachment:image.png)

## ETL Process

### Extraction
- **Hydropower Data (`Data.csv`)**: Approximately 3GB in size, containing ~39 million records. Loaded into a DuckDB database to manage memory usage and enable fast SQL-style querying on disk.
- **Climate Data (`Kalam Climate Data.xlsx`)**: Extracted into DuckDB for consistent handling.
- **Sample Submission**: Used to align and validate the structure of the final forecast output.

### Transformation
- Aggregated both hydropower and climate data to **daily granularity** to align with the forecasting target.
- Created additional features for exploratory data analysis, including lag features and time-based attributes.
- Cleaned and normalized missing or corrupted values where necessary.

### Loading
- The transformed datasets remain within DuckDB for efficient retrieval.
- This also allows easy integration of results (e.g., forecasts) back into the same ecosystem for inspection or downstream usage.

---

## Data Modeling

- **Model Used**: `LinearRegressionModel` from the Darts library.
- **Assumptions**:
  - Energy generation patterns are influenced by historical trends and daily climate variables.
  - A linear relationship (with lags and covariates) is sufficient to capture underlying patterns.
- **Feature Engineering**:
  - Included lagged values of target variables and static covariates from climate data.
  - Normalized data using `StandardScaler` to ensure compatibility with the regression model.
- **Training Process**:
  - Trained on each of the series individually.
  - Evaluation via Root Mean Squared Error (RMSE).
  - Hyperparameter tuning was explored using `Optuna` (not included in the final run for simplicity).
- **Validation**:
  - Cross-validation using historical time slices for each series.
  - Forecast quality was compared visually and numerically using RMSE.

---

## Inference

- **Deployment**: Inference is performed in-batch inside the notebook using Darts’ built-in functionality.
- **New Data Input**: New daily values can be ingested by updating the DuckDB tables and re-running the prediction loop.
- **Forecast Output**: The model outputs a 31-day forecast per time series, which is compiled into a submission-ready format.
- **Model Updates**:
  - Retraining strategy involves re-running the entire pipeline with newly appended data.
  - Versioning of forecasts and inputs can be managed using database snapshots or file versioning.

### Model Equation

$$
\hat{y}_t = \beta_0 + \sum_{i=1}^{k} \beta_i y_{t-i} + \sum_{j=1}^{m} \gamma_j x_{t-j}
$$

Where:
- $ \hat{y}_t $: Predicted power usage at time $ t $
- $ y_{t-i} $: Lagged values of the target series (power usage)
- $ x_{t-j} $: Lagged or static exogenous covariates (e.g. temperature, wind speed, precipitation)
- $ \beta_0 $: Intercept term
- $ \beta_i $, $ \gamma_j $: Coefficients learned during training
- $ k $: Number of lags of the target series used (lookback period or **input/chunk length**)
- $ m $: Number of covariate features (lagged or static)

### Chunk Length

The parameter `input_chunk_length = k` defines how many previous time steps are considered when making a forecast. For example, with `input_chunk_length = 30`, the model uses the last 30 days of power usage and climate data to forecast future values.

This chunk serves as the **sliding window** over which the model "looks back" to make its predictions.

---

## Run Time

- **Notebook Execution Time**: ~54 seconds end-to-end on a machine with 32GB RAM and AMD Ryzen 7 CPU.
- **Heavy Processing**: Primarily during the aggregation and feature engineering phase, mitigated by DuckDB.

---

## Performance Metrics

- **Model Metric**: RMSE was used during validation for each time series.
- **Public Leaderboard Score**: *(insert your score here)*
- **Private Leaderboard Score**: *(insert your score here)*
- **Additional Observations**:
  - Simpler models (e.g., linear regression with lagged features) offered competitive performance due to the high signal-to-noise ratio in temporal patterns.
  - Complex models like deep learning were explored during experimentation but excluded from the final run due to compute constraints and marginal performance gains.