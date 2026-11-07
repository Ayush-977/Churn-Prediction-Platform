import mlflow
import mlflow.sklearn


def log_experiment(
        params,
        metrics,
        model
):

    mlflow.set_experiment(
    "Customer Churn Prediction"
)

    with mlflow.start_run():

        

        for k, v in params.items():
            mlflow.log_param(k, v)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="cloudpickle"
        )