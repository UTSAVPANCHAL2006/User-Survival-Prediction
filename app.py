import uvicorn
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Titanic Survival Prediction")


MODEL_PATH = "artifacts/model"
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

class InputData(BaseModel):
    Pclass: int
    Sex: str
    Age: Optional[float] = None
    SibSp: int
    Parch: int
    Fare: float
    Cabin: Optional[str] = None 
    Embarked: Optional[str] = None

@app.get("/")
def read_root():
    return {"message": "Titanic Survival Prediction API"}

def preprocess_data(data: InputData) -> pd.DataFrame:
  
    df = pd.DataFrame([data.dict()])

    
    if df['Age'].isnull().any():
        df['Age'] = df['Age'].fillna(29.699) 
    
    df['Embarked'] = df['Embarked'].fillna('S')

  
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    df['HasCabin'] = df['Cabin'].notnull().astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    
    expected_cols = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'HasCabin']
    
    X = df[expected_cols]
    
    return X

@app.post("/predict")
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        X = preprocess_data(data)
        prediction = model.predict(X)
        probability = model.predict_proba(X).max() if hasattr(model, "predict_proba") else None
        
        result = int(prediction[0])
        
        return {
            "prediction": result,
            "prediction_label": "Survived" if result == 1 else "Did not survive",
            "probability": float(probability) if probability is not None else None
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
