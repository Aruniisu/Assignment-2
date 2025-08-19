# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

# Load the trained model
model = joblib.load("model.pkl")

# Create FastAPI app
app = FastAPI(title="Iris Classification API", 
              description="API for classifying iris flowers")

# Define input schema
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionOutput(BaseModel):
    species: str
    species_id: int
    confidence: float = None

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Iris Classification API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: IrisFeatures):
    try:
        # Convert input to numpy array
        features = np.array([[input_data.sepal_length, 
                             input_data.sepal_width, 
                             input_data.petal_length, 
                             input_data.petal_width]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0].max()
        
        # Map prediction to species name
        species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        
        return PredictionOutput(
            species=species_map[prediction],
            species_id=prediction,
            confidence=float(proba)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "model_type": "RandomForestClassifier",
        "problem_type": "classification",
        "classes": ["setosa", "versicolor", "virginica"],
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "accuracy": 1.0  # Replace with your actual test accuracy
    }