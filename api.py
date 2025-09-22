# api.py
from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# Load model
model = joblib.load("decision_tree_model.pkl")

# Define the order of features (must match training)
FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

# Request schema
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Initialize API
app = FastAPI(title="Decision Tree API", version="1.0")

@app.post("/predict")
def predict(request: IrisRequest):
    # Map request fields to training feature names
    input_data = pd.DataFrame([[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width
    ]], columns=FEATURE_COLUMNS)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    return {"prediction": int(prediction)}
