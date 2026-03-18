# Customer Churn Prediction (Telco)

End-to-end churn prediction project built on the Kaggle Telco Customer Churn dataset:
blastchar/telco-customer-churn

This repository includes data preparation guidance in Excel, SQL feature engineering with cohort and RFM logic, and a production-style Python training pipeline that saves plots and model artifacts.

## Project Goals

- Predict whether a customer will churn.
- Compare multiple classification models using business-relevant metrics.
- Generate interpretable model diagnostics and feature importance.
- Save reusable artifacts for downstream scoring and reporting.

## Portfolio Value

This project demonstrates end-to-end analytics and machine learning delivery:

- Data preparation in Excel for stakeholder-friendly validation.
- Advanced SQL feature engineering using window functions and cohort segmentation.
- Reproducible Python pipeline with train/test discipline and imbalance handling.
- Multi-model benchmarking with cross-validated hyperparameter search.
- Explainability outputs (SHAP) and business-facing evaluation plots.

Resume-ready impact statement:

Built an end-to-end customer churn prediction system on 7,000+ telco records, combining Excel data QA, PostgreSQL cohort-RFM feature engineering, and tuned Python ML models with SHAP explainability and production model artifacts.

## Repository Structure

- excel/excel_prep_steps.md: Excel cleaning workflow (duplicates, TotalCharges blanks, tenure_group, pivot prep).
- sql/churn_rfm_features.sql: PostgreSQL CTE query with NTILE, LAG, and AVG window functions for RFM and risk segmentation.
- src/train_churn_model.py: Main Python pipeline for EDA, feature engineering, training, evaluation, and artifact export.
- requirements.txt: Python dependencies.
- outputs/plots: Generated visual outputs.
- outputs/models: Saved model artifacts.

## Dataset

Download the dataset from Kaggle and place the CSV at:

data/WA_Fn-UseC_-Telco-Customer-Churn.csv

## Quick Start (Windows PowerShell)

1. Create virtual environment:

```powershell
py -m venv venv
```

2. Activate environment:

```powershell
.\venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Run the full pipeline:

```powershell
python src/train_churn_model.py
```

Optional run modes:

```powershell
# Optimize search by ROC-AUC instead of accuracy
python src/train_churn_model.py --optimize-metric roc_auc

# Use SMOTE on the training split
python src/train_churn_model.py --use-smote
```

## Implemented Workflow

### 1) Excel Preparation

- Removes duplicates by customerID.
- Fixes blank TotalCharges values.
- Adds tenure_group banding column.
- Creates pivot-ready churn summary by contract type.

Reference: excel/excel_prep_steps.md

### 2) SQL Cohort + RFM Features

- Uses CTEs for staged transformation.
- Uses NTILE(5) for R/F/M quantile scores.
- Uses AVG() OVER (PARTITION BY ...) for cohort averages.
- Uses LAG() to create previous-value trend context.
- Outputs risk_segment labels: High Risk, Med Risk, Low Risk.

Reference: sql/churn_rfm_features.sql

### 3) Python EDA and Preprocessing

- Churn distribution chart (shows class imbalance, approx 26.5% churn).
- Churn rate by contract type.
- MonthlyCharges vs Churn boxplot.
- Correlation heatmap for numerical features.
- Feature engineering:
  - avg_monthly_spend
  - is_new_customer
  - multi_services
- One-hot encoding with pandas.get_dummies().
- Stratified 80/20 train-test split.
- SMOTE applied only to training set.

### 4) Model Training and Comparison

Models trained:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- CatBoost (native categorical handling)
- Soft Voting Ensemble (top-3 model probabilities)

Metrics reported for each:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

The script prints a model comparison table sorted by Accuracy and ROC-AUC, and saves both metrics and best hyperparameters.

### 5) Evaluation Artifacts

- Combined ROC curves for all trained models.
- Confusion Matrix for XGBoost.
- Precision-Recall curve for XGBoost.
- SHAP summary plot for XGBoost feature importance.
- Test-set predictions export with risk segmentation.
- ROI scenario table for retention planning.

## Output Files

Generated in outputs/plots:

- 01_churn_distribution.png
- 02_churn_rate_by_contract.png
- 03_monthlycharges_vs_churn_boxplot.png
- 04_correlation_heatmap.png
- 05_roc_curve_all_models.png
- 06_confusion_matrix_xgboost.png
- 07_precision_recall_xgboost.png
- 08_shap_summary_xgboost.png
- pivot_churn_by_contract.csv
- model_comparison_metrics.csv
- model_best_params.csv
- predictions.csv
- roi_calculator.csv

Saved model:

- outputs/models/xgb_churn_model.pkl

## Reproducibility Notes

- Script paths are resolved relative to the repository root, so the pipeline can be launched from any working directory.
- The test set is an 80/20 stratified holdout split on churn labels using random state 42.
- One-hot encoding is applied after train/test split to avoid preprocessing leakage.
- Decision thresholds are selected on a validation split from training data before final test-set scoring.
- The training script validates required input columns before model execution.

## Latest Run Snapshot

- Observed churn rate: 26.54%
- CatBoost Raw: Accuracy 0.8041, ROC-AUC 0.8450
- Soft Voting Ensemble: Accuracy 0.8034, ROC-AUC 0.8467
- Logistic Regression: Accuracy 0.8020, ROC-AUC 0.8437
- XGBoost: Accuracy 0.8013, ROC-AUC 0.8472
- Gradient Boosting: Accuracy 0.7999, ROC-AUC 0.8460
- Random Forest: Accuracy 0.7999, ROC-AUC 0.8420

Target criteria:

- Accuracy >= 0.82
- ROC-AUC > 0.85

Current baseline does not yet meet target; further tuning is recommended.

## Key Business Insights

- Churn remains heavily imbalanced around 26.5 percent, requiring metric selection beyond accuracy.
- Month-to-month contracts show materially higher churn than longer-term contracts.
- MonthlyCharges distribution is generally higher among churned customers.
- Service-adoption combinations and payment behavior improve churn signal.

## Interview Talking Points

- Why stratified split + SMOTE only on train:
  Preserves realistic test distribution while improving minority learning.
- Why ROC-AUC and PR curves:
  Better for imbalanced classification than accuracy alone.
- Why SQL + Python split:
  Mirrors real production analytics stacks where feature logic often lives in the warehouse.

## Suggested Next Improvements

- Hyperparameter optimization for XGBoost (grid/random/Bayesian search).
- Probability threshold tuning for improved precision-recall balance.
- Cross-validation with calibrated probabilities.
- Additional engineered features (interaction terms, contract/payment risk indices).
