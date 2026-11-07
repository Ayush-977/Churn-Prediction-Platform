import pandas as pd
from src.config.config import DROP_COLUMNS

def load_data(filepath):

    df = pd.read_csv(
        filepath,
        encoding="latin1"
    )

    df.drop(
        columns=DROP_COLUMNS,
        inplace=True,
        errors="ignore"
    )

    df["Churn Label"] = df["Churn Label"].map(
        {
            "Yes":1,
            "No":0
        }
    )

    df["Total Charges"] = pd.to_numeric(
        df["Total Charges"],
        errors="coerce"
    )

    return df