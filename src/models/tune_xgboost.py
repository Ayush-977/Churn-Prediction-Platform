import optuna

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier


def tune_xgboost(
        preprocessor,
        X_train,
        y_train,
        X_test,
        y_test
):

    def objective(trial):

        params = {
            "max_depth":
            trial.suggest_int(
                "max_depth",
                3,
                10
            ),

            "learning_rate":
            trial.suggest_float(
                "learning_rate",
                0.01,
                0.3,
                log=True
            ),

            "n_estimators":
            trial.suggest_int(
                "n_estimators",
                100,
                500
            ),

            "subsample":
            trial.suggest_float(
                "subsample",
                0.6,
                1.0
            ),

            "colsample_bytree":
            trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0
            ),

            "random_state":42,
            "eval_metric":"logloss"
        }

        model = XGBClassifier(
            **params
        )

        pipe = Pipeline([
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ])

        pipe.fit(
            X_train,
            y_train
        )

        probs = pipe.predict_proba(
            X_test
        )[:,1]

        return roc_auc_score(
            y_test,
            probs
        )

    study = optuna.create_study(
        direction="maximize"
    )

    study.optimize(
        objective,
        n_trials=30
    )

    return study