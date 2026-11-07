from fastapi import FastAPI

from src.api.schemas import (
    PredictionRequest,
    PredictionResponse
)

from src.api.services import predict


app = FastAPI(
    title="Customer Churn API",
    version="1.0.0"
)


@app.get("/")
def root():

    return {
        "message":
        "Customer Churn Prediction API"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_customer(
    request: PredictionRequest
):

    result = predict(
        request.model_dump()
    )

    return result