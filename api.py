import os
import joblib
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime, timedelta

# ----------------------------
# Load model
# ----------------------------
model = joblib.load("decision_tree_model.pkl")

FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

# ----------------------------
# Load API keys + limits from ENV
# Format: "apikey1:100,apikey2:50"
# ----------------------------
API_KEYS = {}
raw_keys = os.getenv("API_KEYS", "")

for item in raw_keys.split(","):
    if ":" in item:
        key, limit = item.split(":")
        API_KEYS[key.strip()] = {
            "limit": int(limit.strip()),
            "used": 0,
            "reset_time": datetime.utcnow() + timedelta(days=1)
        }

# ----------------------------
# Request schema
# ----------------------------
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# ----------------------------
# Init FastAPI
# ----------------------------
app = FastAPI(title="Decision Tree API with Quotas", version="1.0")

# ----------------------------
# Auth & Quota verification
# ----------------------------
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    user = API_KEYS[x_api_key]

    # Reset if expired
    if datetime.utcnow() > user["reset_time"]:
        user["used"] = 0
        user["reset_time"] = datetime.utcnow() + timedelta(days=1)

    if user["used"] >= user["limit"]:
        raise HTTPException(status_code=429, detail="Daily quota exceeded")

    user["used"] += 1
    return user

# ----------------------------
# Prediction endpoint
# ----------------------------
@app.post("/predict")
def predict(request: IrisRequest, user: dict = Depends(verify_api_key)):
    input_data = pd.DataFrame([[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width
    ]], columns=FEATURE_COLUMNS)

    prediction = model.predict(input_data)[0]
    remaining = user["limit"] - user["used"]

    return {
        "prediction": int(prediction),
        "remaining_quota": remaining,
        "reset_at": user["reset_time"].isoformat()
    }
