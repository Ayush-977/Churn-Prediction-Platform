import shap
import matplotlib.pyplot as plt


def run_shap(
        pipeline,
        X_test
):

    explainer = shap.TreeExplainer(
        pipeline.named_steps["model"]
    )

    X_test_transformed = (
        pipeline.named_steps["preprocessor"]
        .transform(X_test)
    )

    feature_names = (
        pipeline.named_steps["preprocessor"]
        .get_feature_names_out()
    )

    shap_values = explainer.shap_values(
        X_test_transformed
    )

    shap.summary_plot(
        shap_values,
        X_test_transformed,
        feature_names=feature_names,
        show=False
    )

    plt.savefig(
        "artifacts/shap_summary.png",
        bbox_inches="tight"
    )

    plt.close()