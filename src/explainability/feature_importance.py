import pandas as pd

def feature_importance(
        pipeline
):

    feature_names = (
        pipeline.named_steps[
            "preprocessor"
        ].get_feature_names_out()
    )

    importances = (
        pipeline.named_steps[
            "model"
        ].feature_importances_
    )

    feat_imp = pd.DataFrame(
        {
            "feature":
            feature_names,

            "importance":
            importances
        }
    )

    return feat_imp.sort_values(
        "importance",
        ascending=False
    )

import matplotlib.pyplot as plt

def save_feature_importance_plot(feat_imp):

    top20 = feat_imp.head(20)

    plt.figure(figsize=(10,8))

    plt.barh(
        top20["feature"],
        top20["importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 20 Feature Importances")

    plt.tight_layout()

    plt.savefig(
        "artifacts/feature_importance.png"
    )

    plt.close()