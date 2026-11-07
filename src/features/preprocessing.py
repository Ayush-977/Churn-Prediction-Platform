from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


def build_preprocessor(X):

    numerical_cols = X.select_dtypes(
        include=["int64","float64"]
    ).columns

    categorical_cols = X.select_dtypes(
        include=["object"]
    ).columns

    num_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    cat_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                num_pipeline,
                numerical_cols
            ),
            (
                "cat",
                cat_pipeline,
                categorical_cols
            )
        ]
    )

    return preprocessor