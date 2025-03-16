# ev-battery-failure-prediction

End-to-end data science project for predicting battery failure in electrified vehicle fleets using machine learning. Covers data cleaning, feature engineering, modeling, and evaluation to optimize predictive maintenance and fleet performance.

This project shows how battery failure can be predicted using a XGBoost classifier from scikit-learn & scikit-optimize for hyperparameter tuning. The model is trained on a dataset that includes features: temperature, humidity, battery_level, and device_status (categorical), with a binary target variable (`is_outage`).

## Project Structure

- `battery_failure.ipynb`: Jupyter Notebook containing the data analysis, model training, hyperparameter tuning, and evaluation.
- `environment.yml`: Conda environment file to set up all necessary dependencies.
- `requirements.txt`: Pip requirements file for installing dependencies via pip.
- `README.md`: current file.
- `images/`: directory to store reference as well as generated images.
- `models/`: directory to store pickled models.

## Project Overview
1. Data Preprocessing:
- The dataset is split into training, validation, and test sets. 
2. Exploratory Data Analysis (EDA )
- The data is statistically analyzed to detemine what steps to take to get a performant model built.
- To determine feature importance, a RandomForest model is used with stratification to maintain the distribution of the target variable, as it was dtermined to be 30%-70% outtage.

3. Model Training:
- A XGBoost classifier is used, and Bayesian optimization (via BayesSearchCV in sklearn) tunes the learning rate and tree depth. The model is evaluated using cross-validation, with accuracy, precision, recall, and f1-score as metrics.

4. Results:
- The model achieved high accuracy and robust performance on both the validation and test sets. The ROC curve shows an excellent fit. This suggests it is effective at predicting battery outages.
  - <img src="images/metrics_table.png" alt="Fit Scores" width="600" height="200">
  - <img src="images/validation_report.png" alt="Validation Report" width="600" height="200">
  - <img src="images/test_report.png" alt="Test Report" width="600" height="200">
  - <img src="images/roc_curve.png" alt="Test Report" width="600" height="600">


## Setup Instructions

### Using Conda
1. Create the conda environment & activate:
   ```bash
   conda env create -f environment.yml
   conda activate battery-failure
   ```

### Using venv
1. python -m venv venv
   ```bash
      source venv/bin/activate  # (for Windows: venv\Scripts\activate)
   ```
2. install libraries:
   ```bash
      pip install -r requirements.txt
   ```



