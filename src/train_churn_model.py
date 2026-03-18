import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
PLOTS_DIR = Path("outputs/plots")
MODELS_DIR = Path("outputs/models")
OPTIMIZE_METRIC = "accuracy"
USE_SMOTE_FOR_TRAINING = False


def ensure_output_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def clean_and_prepare_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Download from Kaggle (blastchar/telco-customer-churn) and place it there."
        )

    df = pd.read_csv(data_path)
    df = df.drop_duplicates(subset=["customerID"]).copy()

    # Fix blank TotalCharges values and enforce numeric dtype.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Tenure cohorts for retention analysis.
    bins = [-1, 12, 24, 48, 60, np.inf]
    labels = ["0-12 Months", "13-24 Months", "25-48 Months", "49-60 Months", "60+ Months"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels)

    return df


def save_churn_contract_pivot(df: pd.DataFrame) -> None:
    pivot = pd.pivot_table(
        df,
        index="Contract",
        columns="Churn",
        values="customerID",
        aggfunc="count",
        fill_value=0,
    )
    pivot["churn_rate"] = pivot.get("Yes", 0) / pivot.sum(axis=1)
    pivot.to_csv(PLOTS_DIR / "pivot_churn_by_contract.csv")


def plot_eda(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    # 1) Churn distribution
    churn_counts = df["Churn"].value_counts().reindex(["No", "Yes"])
    churn_rate = (churn_counts["Yes"] / churn_counts.sum()) * 100
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=churn_counts.index, y=churn_counts.values, palette=["#4C78A8", "#F58518"])
    ax.set_title(f"Churn Distribution (Yes = {churn_rate:.1f}%)")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Customer Count")
    for i, v in enumerate(churn_counts.values):
        ax.text(i, v + max(churn_counts.values) * 0.01, f"{v:,}", ha="center")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_churn_distribution.png", dpi=300)
    plt.close()

    # 2) Churn rate by contract type
    contract_churn = (
        df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean() * 100).sort_values(ascending=False)
    )
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(x=contract_churn.index, y=contract_churn.values, palette="viridis")
    ax.set_title("Churn Rate by Contract Type")
    ax.set_xlabel("Contract Type")
    ax.set_ylabel("Churn Rate (%)")
    for i, v in enumerate(contract_churn.values):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_churn_rate_by_contract.png", dpi=300)
    plt.close()

    # 3) Monthly charges vs churn boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", palette=["#4C78A8", "#F58518"])
    plt.title("Monthly Charges vs Churn")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_monthlycharges_vs_churn_boxplot.png", dpi=300)
    plt.close()

    # 4) Correlation heatmap for numerical features
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    corr = df[num_cols].corr()
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Correlation Heatmap (Numerical Features)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_correlation_heatmap.png", dpi=300)
    plt.close()

    print(f"Observed churn rate: {churn_rate:.2f}% (benchmark ~26.5%)")


def feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df_fe = df.copy()

    df_fe["avg_monthly_spend"] = np.where(
        df_fe["tenure"] > 0,
        df_fe["TotalCharges"] / df_fe["tenure"],
        df_fe["MonthlyCharges"],
    )
    df_fe["is_new_customer"] = (df_fe["tenure"] <= 12).astype(int)
    df_fe["tenure_charge_ratio"] = df_fe["tenure"] / (df_fe["MonthlyCharges"] + 1)
    df_fe["charge_per_tenure_plus1"] = df_fe["TotalCharges"] / (df_fe["tenure"] + 1)
    df_fe["service_intensity"] = df_fe["MonthlyCharges"] / (df_fe["avg_monthly_spend"] + 1)
    df_fe["senior_and_new"] = ((df_fe["SeniorCitizen"] == 1) & (df_fe["is_new_customer"] == 1)).astype(int)
    df_fe["electronic_check_flag"] = (df_fe["PaymentMethod"] == "Electronic check").astype(int)
    df_fe["auto_pay_flag"] = (
        df_fe["PaymentMethod"].str.contains("automatic", case=False, na=False)
    ).astype(int)
    df_fe["partner_dependents_flag"] = (
        (df_fe["Partner"] == "Yes") & (df_fe["Dependents"] == "Yes")
    ).astype(int)

    service_cols = [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    existing_service_cols = [c for c in service_cols if c in df_fe.columns]
    df_fe["multi_services"] = (
        df_fe[existing_service_cols]
        .replace({"No internet service": "No", "No phone service": "No"})
        .apply(lambda col: (col == "Yes").astype(int))
        .sum(axis=1)
    )

    y = (df_fe["Churn"] == "Yes").astype(int)
    X = df_fe.drop(columns=["customerID", "Churn"])

    return X, y


def encode_train_test_features(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # One-hot encode after split to avoid train-test preprocessing leakage.
    cat_cols_train = X_train_raw.select_dtypes(include=["object", "category"]).columns
    cat_cols_test = X_test_raw.select_dtypes(include=["object", "category"]).columns
    categorical_cols = sorted(set(cat_cols_train).union(set(cat_cols_test)))

    X_train_encoded = pd.get_dummies(X_train_raw, columns=categorical_cols, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test_raw, columns=categorical_cols, drop_first=True)

    X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join="left", axis=1, fill_value=0)
    return X_train_encoded, X_test_encoded


def find_best_threshold(y_true: pd.Series, y_proba: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.20, 0.96, 0.01)
    accuracies = [accuracy_score(y_true, (y_proba >= thr).astype(int)) for thr in thresholds]
    best_idx = int(np.argmax(accuracies))
    return float(thresholds[best_idx]), float(accuracies[best_idx])


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict, dict]:
    model_spaces = {
        "Logistic Regression": {
            "estimator": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
                ]
            ),
            "params": {
                "model__C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                "model__solver": ["liblinear", "lbfgs"],
                "model__class_weight": [None, "balanced"],
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
            "params": {
                "n_estimators": [100, 200, 300, 400, 500],
                "max_depth": [None, 5, 8, 12, 16],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
            },
        },
        "Gradient Boosting": {
            "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "params": {
                "n_estimators": [100, 200, 300, 400],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "max_depth": [2, 3, 4],
                "subsample": [0.7, 0.85, 1.0],
                "min_samples_leaf": [1, 2, 4],
            },
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": [200, 300, 400, 500, 700],
                "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
                "max_depth": [3, 4, 5, 6],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
                "min_child_weight": [1, 3, 5],
                "gamma": [0.0, 0.1, 0.2],
                "reg_lambda": [1.0, 2.0, 5.0],
            },
        },
    }

    rows = []
    roc_data = {}
    fitted_models = {}
    best_params = {}
    best_cv_scores = {}
    best_thresholds = {}
    best_oof_accuracy = {}
    train_prob_map = {}
    test_prob_map = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    n_iter_map = {
        "Logistic Regression": 16,
        "Random Forest": 24,
        "Gradient Boosting": 24,
        "XGBoost": 32,
    }

    for name, config in model_spaces.items():
        search = RandomizedSearchCV(
            estimator=config["estimator"],
            param_distributions=config["params"],
            n_iter=n_iter_map[name],
            scoring=OPTIMIZE_METRIC,
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_train_proba = best_model.predict_proba(X_train)[:, 1]
        best_threshold, best_train_acc = find_best_threshold(y_train, y_train_proba)

        y_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= best_threshold).astype(int)

        fpr, tpr, _ = roc_curve(y_test, y_proba)

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
            "Best Threshold": best_threshold,
            "Train Accuracy": best_train_acc,
            "CV Accuracy": search.best_score_,
        }
        rows.append(metrics)
        roc_data[name] = (fpr, tpr, auc(fpr, tpr))
        fitted_models[name] = best_model
        best_params[name] = search.best_params_
        best_cv_scores[name] = search.best_score_
        best_thresholds[name] = best_threshold
        best_oof_accuracy[name] = best_train_acc
        train_prob_map[name] = y_train_proba
        test_prob_map[name] = y_proba

    top_models = sorted(best_oof_accuracy, key=best_oof_accuracy.get, reverse=True)[:3]
    ensemble_train_proba = np.mean([train_prob_map[m] for m in top_models], axis=0)
    ensemble_test_proba = np.mean([test_prob_map[m] for m in top_models], axis=0)
    ensemble_thresholds = np.arange(0.25, 0.81, 0.01)
    ensemble_train_acc = [
        accuracy_score(y_train, (ensemble_train_proba >= thr).astype(int)) for thr in ensemble_thresholds
    ]
    ensemble_best_idx = int(np.argmax(ensemble_train_acc))
    ensemble_best_threshold = float(ensemble_thresholds[ensemble_best_idx])
    ensemble_pred = (ensemble_test_proba >= ensemble_best_threshold).astype(int)
    ensemble_fpr, ensemble_tpr, _ = roc_curve(y_test, ensemble_test_proba)

    rows.append(
        {
            "Model": "Soft Voting Ensemble",
            "Accuracy": accuracy_score(y_test, ensemble_pred),
            "Precision": precision_score(y_test, ensemble_pred, zero_division=0),
            "Recall": recall_score(y_test, ensemble_pred, zero_division=0),
            "F1": f1_score(y_test, ensemble_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, ensemble_test_proba),
            "Best Threshold": ensemble_best_threshold,
            "Train Accuracy": ensemble_train_acc[ensemble_best_idx],
            "CV Accuracy": float(np.mean([best_cv_scores[m] for m in top_models])),
        }
    )
    roc_data["Soft Voting Ensemble"] = (
        ensemble_fpr,
        ensemble_tpr,
        auc(ensemble_fpr, ensemble_tpr),
    )
    best_params["Soft Voting Ensemble"] = {"members": top_models}
    best_thresholds["Soft Voting Ensemble"] = ensemble_best_threshold
    best_oof_accuracy["Soft Voting Ensemble"] = ensemble_train_acc[ensemble_best_idx]
    best_cv_scores["Soft Voting Ensemble"] = float(np.mean([best_cv_scores[m] for m in top_models]))

    results_df = (
        pd.DataFrame(rows)
        .sort_values(by=["Accuracy", "ROC-AUC"], ascending=False)
        .reset_index(drop=True)
    )
    results_df.to_csv(PLOTS_DIR / "model_comparison_metrics.csv", index=False)
    params_df = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Best_Threshold": best_thresholds[model_name],
                "Train_Accuracy_At_Best_Threshold": best_oof_accuracy[model_name],
                "Best_CV_Accuracy": best_cv_scores[model_name],
                "Best_Params": str(params),
            }
            for model_name, params in best_params.items()
        ]
    )
    params_df.to_csv(PLOTS_DIR / "model_best_params.csv", index=False)

    print("\nBest hyperparameters by model:")
    for model_name, params in best_params.items():
        print(f"- {model_name}: {params}")

    return results_df, roc_data, fitted_models


def train_and_evaluate_catboost_raw(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    X_test_raw: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict, tuple[np.ndarray, np.ndarray, float], CatBoostClassifier, dict]:
    cat_cols = list(X_train_raw.select_dtypes(include=["object", "category"]).columns)

    X_train_cb = X_train_raw.copy()
    X_test_cb = X_test_raw.copy()
    for col in cat_cols:
        X_train_cb[col] = X_train_cb[col].astype(str)
        X_test_cb[col] = X_test_cb[col].astype(str)

    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
        X_train_cb,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    estimator = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0)
    params = {
        "depth": [4, 5, 6, 7, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.08],
        "iterations": [200, 400, 600, 800],
        "l2_leaf_reg": [1, 3, 5, 7, 9],
        "border_count": [64, 128, 254],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=params,
        n_iter=24,
        scoring=OPTIMIZE_METRIC,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train_sub, y_train_sub, cat_features=cat_cols)

    best_model = search.best_estimator_
    y_val_proba = best_model.predict_proba(X_val_sub)[:, 1]
    best_threshold, best_val_acc = find_best_threshold(y_val_sub, y_val_proba)

    best_model.fit(X_train_cb, y_train, cat_features=cat_cols)
    y_test_proba = best_model.predict_proba(X_test_cb)[:, 1]
    y_test_pred = (y_test_proba >= best_threshold).astype(int)

    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    metrics = {
        "Model": "CatBoost Raw",
        "Accuracy": accuracy_score(y_test, y_test_pred),
        "Precision": precision_score(y_test, y_test_pred, zero_division=0),
        "Recall": recall_score(y_test, y_test_pred, zero_division=0),
        "F1": f1_score(y_test, y_test_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_test_proba),
        "Best Threshold": best_threshold,
        "Train Accuracy": best_val_acc,
        "CV Accuracy": search.best_score_,
    }
    meta = {
        "Best_Threshold": best_threshold,
        "Train_Accuracy_At_Best_Threshold": best_val_acc,
        "Best_CV_Accuracy": search.best_score_,
        "Best_Params": str(search.best_params_),
        "cat_features": cat_cols,
    }

    return metrics, (fpr, tpr, auc(fpr, tpr)), best_model, meta


def plot_roc_curves(roc_data: dict) -> None:
    plt.figure(figsize=(9, 6))
    for name, (fpr, tpr, roc_auc) in roc_data.items():
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_roc_curve_all_models.png", dpi=300)
    plt.close()


def plot_confusion_matrix_best(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues")
    disp.ax_.set_title("Confusion Matrix - XGBoost (0.50 Threshold)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_confusion_matrix_xgboost.png", dpi=300)
    plt.close()


def plot_precision_recall_best(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, color="#F58518")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - XGBoost")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_precision_recall_xgboost.png", dpi=300)
    plt.close()


def plot_shap_summary(model, X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    sample_size = min(1000, len(X_test))
    X_test_sample = X_test.sample(n=sample_size, random_state=RANDOM_STATE)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "08_shap_summary_xgboost.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    ensure_output_dirs()

    df = clean_and_prepare_data(DATA_PATH)
    save_churn_contract_pivot(df)
    plot_eda(df)

    X_raw, y = feature_engineering(df)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_test = encode_train_test_features(X_train_raw, X_test_raw)

    if USE_SMOTE_FOR_TRAINING:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_fit, y_train_fit = smote.fit_resample(X_train, y_train)
    else:
        X_train_fit, y_train_fit = X_train, y_train

    results_df, roc_data, fitted_models = train_and_evaluate_models(
        X_train_fit,
        y_train_fit,
        X_test,
        y_test,
    )

    cat_raw_metrics, cat_raw_roc, cat_raw_model, cat_raw_meta = train_and_evaluate_catboost_raw(
        X_train_raw,
        y_train,
        X_test_raw,
        y_test,
    )
    results_df = pd.concat([results_df, pd.DataFrame([cat_raw_metrics])], ignore_index=True)
    results_df = results_df.sort_values(by=["Accuracy", "ROC-AUC"], ascending=False).reset_index(drop=True)
    roc_data["CatBoost Raw"] = cat_raw_roc
    fitted_models["CatBoost Raw"] = cat_raw_model
    results_df.to_csv(PLOTS_DIR / "model_comparison_metrics.csv", index=False)

    params_df = pd.read_csv(PLOTS_DIR / "model_best_params.csv")
    params_df = pd.concat(
        [
            params_df,
            pd.DataFrame(
                [
                    {
                        "Model": "CatBoost Raw",
                        "Best_Threshold": cat_raw_meta["Best_Threshold"],
                        "Train_Accuracy_At_Best_Threshold": cat_raw_meta[
                            "Train_Accuracy_At_Best_Threshold"
                        ],
                        "Best_CV_Accuracy": cat_raw_meta["Best_CV_Accuracy"],
                        "Best_Params": cat_raw_meta["Best_Params"],
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    params_df.to_csv(PLOTS_DIR / "model_best_params.csv", index=False)

    print("\nModel Comparison (sorted by Accuracy, then ROC-AUC):")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    plot_roc_curves(roc_data)

    xgb_model = fitted_models["XGBoost"]
    plot_confusion_matrix_best(xgb_model, X_test, y_test)
    plot_precision_recall_best(xgb_model, X_test, y_test)
    plot_shap_summary(xgb_model, X_train_fit, X_test)

    model_artifact = {
        "model": xgb_model,
        "feature_columns": list(X_train.columns),
        "random_state": RANDOM_STATE,
        "optimization_metric": OPTIMIZE_METRIC,
        "use_smote_for_training": USE_SMOTE_FOR_TRAINING,
        "catboost_raw_meta": cat_raw_meta,
    }
    joblib.dump(model_artifact, MODELS_DIR / "xgb_churn_model.pkl")

    xgb_metrics = results_df[results_df["Model"] == "XGBoost"].iloc[0]
    if xgb_metrics["Accuracy"] >= 0.82 and xgb_metrics["ROC-AUC"] > 0.85:
        print("\nTarget achieved: Accuracy >= 82% and ROC-AUC > 0.85")
    else:
        print("\nTarget not met yet. Consider hyperparameter tuning and feature selection.")

    print(f"\nSaved plots to: {PLOTS_DIR.resolve()}")
    print(f"Saved model to: {(MODELS_DIR / 'xgb_churn_model.pkl').resolve()}")


if __name__ == "__main__":
    main()


  # ── PREDICTIONS EXPORT FOR EXCEL ──────────────────────────────
# Add this right after: joblib.dump(model_artifact, ...)

# Get XGBoost predictions on test set
xgb_threshold = results_df[results_df['Model'] == 'XGBoost']['Best Threshold'].values[0]
y_prob_xgb    = xgb_model.predict_proba(X_test)[:, 1]
y_pred_xgb    = (y_prob_xgb >= xgb_threshold).astype(int)

# Build predictions dataframe
predictions_df = pd.DataFrame({
    'Actual':           y_test.values,
    'Predicted':        y_pred_xgb,
    'Churn_Probability': np.round(y_prob_xgb, 4),
    'Risk_Segment':     pd.cut(
                            y_prob_xgb,
                            bins=[0, 0.30, 0.50, 0.70, 1.0],
                            labels=['Low Risk', 'Med-Low Risk', 'Med-High Risk', 'High Risk']
                        )
})
predictions_df['Actual_Label']    = predictions_df['Actual'].map({0: 'No Churn', 1: 'Churn'})
predictions_df['Predicted_Label'] = predictions_df['Predicted'].map({0: 'No Churn', 1: 'Churn'})
predictions_df['Correct']         = (predictions_df['Actual'] == predictions_df['Predicted'])

predictions_df.to_csv(PLOTS_DIR / 'predictions.csv', index=False)

# ── ROI CALCULATOR DATA ────────────────────────────────────────
total_customers     = len(y_test) + len(y_train)   # full dataset
avg_monthly_revenue = 64.76
high_risk_count     = (y_prob_xgb >= 0.70).sum()   # threshold for "High Risk"
high_risk_pct       = high_risk_count / len(y_prob_xgb) * 100

roi_data = []
for retention in [0.20, 0.30, 0.40]:
    roi_data.append({
        'Retention_Rate':         f'{int(retention*100)}%',
        'High_Risk_Customers':    high_risk_count,
        'Avg_Monthly_Revenue':    avg_monthly_revenue,
        'Monthly_Revenue_Saved':  round(high_risk_count * avg_monthly_revenue * retention, 2),
        'Annual_Revenue_Saved':   round(high_risk_count * avg_monthly_revenue * retention * 12, 2)
    })

roi_df = pd.DataFrame(roi_data)
roi_df.to_csv(PLOTS_DIR / 'roi_calculator.csv', index=False)

# ── PRINT SUMMARY ─────────────────────────────────────────────
print("\n" + "="*55)
print("EXCEL EXPORT SUMMARY")
print("="*55)
print(f"predictions.csv rows     : {len(predictions_df)}")
print(f"High Risk (prob >= 0.70) : {high_risk_count} customers ({high_risk_pct:.1f}%)")
print(f"\nROI SCENARIOS:")
print(roi_df.to_string(index=False))
print(f"\nSaved to: {PLOTS_DIR.resolve()}")

