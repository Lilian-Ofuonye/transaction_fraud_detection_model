# transaction_fraud_detection_model
![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## Project summary

This project builds an end-to-end machine learning pipeline to detect potentially fraudulent bank transactions in near real time.

The business challenge is practical: reduce fraud losses without blocking too many genuine customers. The notebook takes the project from raw transaction data through data quality checks, exploratory analysis, feature engineering, model comparison, hyperparameter tuning, final evaluation, and explainability.

## Why this project matters

Fraud detection is not just a classification task. A useful model needs to:

- catch high-risk transactions early;
- handle class imbalance;
- reduce false positives;
- avoid data leakage;
- produce explanations that risk, fraud, and compliance teams can understand.

This project focuses on that full workflow rather than only chasing a single accuracy score.

## Dataset

The dataset contains **50,000 transaction records** from a synthetic banking fraud dataset.

Target variable:

| Label | Meaning |
|---|---|
| `0` | Non-fraud transaction |
| `1` | Fraud transaction |

Dataset coverage:

- **50,000** transactions
- **8,963** unique users
- Time period: **1 January 2023 to 31 December 2023**
- Fraud prevalence: **32.1%**
- No missing values
- No duplicate rows

Source: [Kaggle Fraud Detection Transactions Dataset](https://www.kaggle.com/datasets/samayashar/fraud-detection-transactions-dataset)

## Business objective

Build a fraud detection model that can rank transactions by fraud risk and support an operational review process.

The model is designed to answer:

> “Which transactions are most likely to be fraudulent, and what signals drove that decision?”

## Project workflow

1. Loaded and inspected the transaction dataset.
2. Checked missing values, duplicate records, class balance, feature types, and logical consistency.
3. Performed exploratory analysis on monetary, behavioural, categorical, temporal, and risk-related features.
4. Engineered behavioural and time-based fraud indicators.
5. Used a time-aware train/test split to reduce leakage risk.
6. Built preprocessing pipelines for numerical and categorical features.
7. Compared multiple classification models using 5-fold cross-validation.
8. Tuned the best model using `RandomizedSearchCV`.
9. Evaluated final model performance on the untouched test set.
10. Interpreted model decisions using feature importance and SHAP.

## Feature engineering

Key engineered features included:

| Feature | Purpose |
|---|---|
| `Time_Since_Last_Transaction` | Measures how quickly a user makes another transaction |
| `Failure_Rate_7d` | Captures the share of recent failed transactions |
| `High_Risk_Behaviour` | Flags transactions with high recent failures and elevated risk |
| `High_Fraud_time_gap` | Flags short time gaps linked with higher fraud rates |
| `High_fraud_hour` | Flags hours with slightly higher fraud rates |
| `High_Fraud_days_of_week` | Flags days with slightly higher fraud rates |
| `Month` | Captures seasonal transaction patterns |

Identifiers such as `Transaction_ID` and `User_ID` were removed from the final modelling pipeline to reduce memorisation risk.

## Models tested

The project compared five models:

| Model | Mean CV ROC-AUC |
|---|---:|
| Dummy baseline | 0.500 |
| Logistic Regression | 0.802 |
| Support Vector Machine | 0.815 |
| XGBoost | 0.875 |
| Random Forest | 0.877 |

Random Forest and XGBoost performed best, suggesting that non-linear models were better suited to the behavioural fraud patterns in the dataset.

## Final model

The final model is a tuned **Random Forest Classifier** inside a scikit-learn pipeline.

Best hyperparameters:

```text
n_estimators: 200
max_depth: 20
min_samples_split: 2
min_samples_leaf: 4
max_features: log2
```
## Final test performance

| Metric | Score |
|---|---:|
| ROC-AUC | 0.884 |
| Accuracy | 0.879 |
| Precision, fraud class | 0.883 |
| Recall, fraud class | 0.720 |
| F1-score, fraud class | 0.793 |

Classification report:

```text
              precision    recall  f1-score   support

Non-Fraud        0.88      0.95      0.91      6,772
Fraud            0.88      0.72      0.79      3,228

accuracy                             0.88     10,000
macro avg        0.88      0.84      0.85     10,000
weighted avg     0.88      0.88      0.88     10,000
```

Confusion matrix summary:

| Outcome | Count |
|---|---:|
| Correctly classified non-fraud transactions | 6,465 |
| Legitimate transactions incorrectly flagged | 307 |
| Correctly detected fraud transactions | 2,324 |
| Missed fraud transactions | 904 |

## Tuning impact

Hyperparameter tuning improved the model’s ability to catch fraud.

| Model version | CV ROC-AUC | Test ROC-AUC | Accuracy | Fraud Precision | Fraud Recall | Fraud F1 |
|---|---:|---:|---:|---:|---:|---:|
| Before tuning | 0.8775 | 0.8850 | 0.8776 | 0.9434 | 0.6605 | 0.7770 |
| After tuning | 0.8796 | 0.8841 | 0.8789 | 0.8833 | 0.7200 | 0.7933 |

The tuned model accepts a controlled increase in false positives to catch more fraudulent transactions. For fraud teams, that trade-off is often more useful than a model with high precision but lower fraud recall.

## Explainability

The model includes feature importance and SHAP-based analysis to make the fraud predictions easier to audit.

Top model drivers:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `Failed_Transaction_Count_7d` | 50.3% |
| 2 | `Failure_Rate_7d` | 12.1% |
| 3 | `High_Risk_Behaviour` | 9.4% |
| 4 | `Transaction_Distance` | 3.3% |
| 5 | `Transaction_Amount` log-transformed | 3.3% |

The top three features explain **71.8%** of model importance, showing that the model is mainly driven by recent transaction failure behaviour rather than arbitrary noise.

## Key insights

- Recent failed transaction behaviour was the strongest signal of fraud.
- Short time gaps between transactions showed higher fraud rates.
- Transaction amount alone was not enough to separate fraud from non-fraud.
- Non-linear models outperformed linear models, suggesting fraud patterns depend on feature interactions.
- Class weighting helped the model pay more attention to the fraud class without resampling the data.

## Tech stack

- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- XGBoost
- SHAP
- statsmodels

## Repository structure

```text
.
├── fraud_detection1.ipynb        # Main notebook
├── fraud_detection_dataset.csv   # Dataset file, download from Kaggle
├── feature_importance.png        # Generated feature importance chart
├── roc_curve_comparison.png      # Generated ROC comparison chart
```
## How to run the project

1. Clone this repository.

```bash
git clone https://github.com/your-username/fraud-detection-prediction-model.git
cd fraud-detection-prediction-model
```

2. Install the required Python packages.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap statsmodels
```

3. Download the dataset from Kaggle and place it in the project root as:

```text
fraud_detection_dataset.csv
```

4. Run the notebook.

```bash
jupyter notebook fraud_detection1.ipynb
```
## What this project demonstrates

This project demonstrates practical machine learning skills that are relevant to fraud, banking, insurance, and risk analytics roles:

- structured data quality assessment;
- leakage-aware model validation;
- feature engineering from customer behaviour and transaction history;
- supervised classification on imbalanced data;
- model comparison using ROC-AUC;
- hyperparameter tuning;
- clear performance reporting;
- explainable AI using feature importance and SHAP.

## Limitations and next steps

This project is strong as a modelling case study, but the next iteration should include:

- threshold optimisation based on business cost of false positives vs missed fraud;
- precision-recall curve analysis;
- model calibration for better probability estimates;
- a stricter time-series validation strategy;
- deployment as an API for real-time scoring;
- monitoring for model drift and changing fraud patterns.

## Project takeaway

The final model achieves strong fraud-ranking performance with a **0.884 ROC-AUC** and detects **72% of fraudulent transactions** on the test set.

The biggest strength of the project is not just the final score. It is the full modelling workflow: data checks, business framing, feature engineering, model comparison, tuning, and explainability.

