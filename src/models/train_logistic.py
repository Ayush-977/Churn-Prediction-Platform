from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def train_logistic(
        preprocessor,
        X_train,
        y_train
):

    pipeline = Pipeline(
        [
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000
                )
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    return pipeline