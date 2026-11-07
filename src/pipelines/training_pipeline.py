import joblib
import pandas as pd
import mlflow
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src.data.load_data import load_data

from src.features.preprocessing import (
    build_preprocessor
)

from src.models.train_logistic import (
    train_logistic
)

from src.models.train_xgboost import (
    train_xgboost
)

from src.models.tune_xgboost import (
    tune_xgboost
)

from src.models.evaluate import (
    evaluate_model
)

from src.explainability.feature_importance import (
    feature_importance
)

from src.explainability.shap_analysis import (
    run_shap
)

from src.tracking.mlflow_tracker import (
    log_experiment
)


def run_training_pipeline():


    df = load_data(
        "data/Telco_customer_churn.csv"
    )

    X = df.drop(
        [
            "Churn Label",
            "Churn Value",
            "Churn Score",
            "CLTV",
            "Churn Reason"
        ],
        axis=1
    )

    y = df["Churn Label"]



    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
    )



    preprocessor = build_preprocessor(
        X_train
    )



    lr_pipeline = train_logistic(
        preprocessor,
        X_train,
        y_train
    )

    lr_metrics = evaluate_model(
        lr_pipeline,
        X_test,
        y_test
    )

    print("\n===== LOGISTIC REGRESSION =====")
    print(lr_metrics)



    xgb_pipeline = train_xgboost(
        preprocessor,
        X_train,
        y_train
    )

    xgb_metrics = evaluate_model(
        xgb_pipeline,
        X_test,
        y_test
    )

    print("\n===== XGBOOST =====")
    print(xgb_metrics)


    study = tune_xgboost(
        preprocessor,
        X_train,
        y_train,
        X_test,
        y_test
    )

    print("\n===== BEST PARAMS =====")
    print(study.best_params)

    print(
        "Best ROC-AUC:",
        study.best_value
    )


    best_model = XGBClassifier(
        **study.best_params,
        random_state=42,
        eval_metric="logloss"
    )

    best_pipeline = Pipeline(
        [
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                best_model
            )
        ]
    )

    best_pipeline.fit(
        X_train,
        y_train
    )



    final_metrics = evaluate_model(
        best_pipeline,
        X_test,
        y_test
    )

    print("\n===== FINAL MODEL =====")
    print(final_metrics)


    feat_imp = feature_importance(
        best_pipeline
    )

    feat_imp.to_csv(
        "artifacts/top_features.csv",
        index=False
    )

    top20 = feat_imp.head(20)

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        top20["feature"],
        top20["importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(
        "Top 20 Feature Importances"
    )

    plt.tight_layout()

    plt.savefig(
        "artifacts/feature_importance.png"
    )

    plt.close()


    run_shap(
        best_pipeline,
        X_test
    )



    comparison = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "XGBoost"
            ],
            "Accuracy": [
                lr_metrics["accuracy"],
                xgb_metrics["accuracy"]
            ],
            "F1": [
                lr_metrics["f1"],
                xgb_metrics["f1"]
            ],
            "ROC_AUC": [
                lr_metrics["roc_auc"],
                xgb_metrics["roc_auc"]
            ]
        }
    )

    comparison.to_csv(
        "artifacts/model_comparison.csv",
        index=False
    )


    log_experiment(
        study.best_params,
        final_metrics,
        best_pipeline
    )


    joblib.dump(
        best_pipeline,
        "artifacts/best_churn_model.pkl"
    )

    print(
        "\nBest Model Saved Successfully!"
    )