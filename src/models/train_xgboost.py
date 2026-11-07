from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

def train_xgboost(
        preprocessor,
        X_train,
        y_train
):

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss"
            )
        )
    ])

    pipeline.fit(
        X_train,
        y_train
    )

    return pipeline