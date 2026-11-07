from pydantic import BaseModel


class PredictionRequest(BaseModel):
    Gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    TenureMonths: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictionResponse(BaseModel):

    prediction: int
    churn_probability: float
    risk_level: str
    recommendation: str