import joblib
import pandas as pd


model = joblib.load(
    "artifacts/best_churn_model.pkl"
)

COLUMN_MAPPING = {
    "SeniorCitizen": "Senior Citizen",
    "TenureMonths": "Tenure Months",
    "PhoneService": "Phone Service",
    "MultipleLines": "Multiple Lines",
    "InternetService": "Internet Service",
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection",
    "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
    "PaperlessBilling": "Paperless Billing",
    "PaymentMethod": "Payment Method",
    "MonthlyCharges": "Monthly Charges",
    "TotalCharges": "Total Charges"
}


def get_recommendation(prob):

    if prob > 0.8:
        return "Offer 20% discount"

    elif prob > 0.6:
        return "Offer loyalty rewards"

    elif prob > 0.4:
        return "Customer engagement campaign"

    return "Low Risk Customer"


def predict(data):


    data = {
        COLUMN_MAPPING.get(k, k): v
        for k, v in data.items()
    }

    df = pd.DataFrame([data])

    probability = float(
        model.predict_proba(df)[0][1]
    )

    prediction = int(
        probability >= 0.5
    )

    if probability > 0.8:
        risk = "High"

    elif probability > 0.5:
        risk = "Medium"

    else:
        risk = "Low"

    return {
        "prediction": prediction,
        "churn_probability": round(
            probability,
            4
        ),
        "risk_level": risk,
        "recommendation":
            get_recommendation(
                probability
            )
    }