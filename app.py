import uvicorn
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from typing import Optional
from pathlib import Path
import joblib



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = joblib.load("artifacts/model")

AGE_MEAN = 29.699
SEX_MAP = {"male": 0, "female": 1}
EMBARKED_MAP = {"C": 0, "Q": 1, "S": 2}

EXPECTED_COLUMNS = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Embarked",
    "FamilySize",
    "HasCabin",
]

try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    logger.info(f"✅ Model loaded successfully from {MODEL_PATH}")

except Exception as e:
    logger.critical(f"❌ Failed to load model: {e}")
    model = None


app = FastAPI(
    title="Titanic Survival Prediction API",
    version="1.0.0"
)


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
def root():
    return {"message": "🚢 Titanic Survival Prediction API is running"}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


def preprocess_data(data: InputData) -> pd.DataFrame:
    df = pd.DataFrame([data.dict()])

    
    df["Age"] = df["Age"].fillna(AGE_MEAN)
    df["Embarked"] = df["Embarked"].fillna("S")

    df["Sex"] = df["Sex"].map(SEX_MAP)
    df["Embarked"] = df["Embarked"].map(EMBARKED_MAP)

    if df["Sex"].isnull().any() or df["Embarked"].isnull().any():
        raise ValueError("Invalid category in Sex or Embarked")

    df["HasCabin"] = df["Cabin"].notnull().astype(int)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    return df[EXPECTED_COLUMNS]



@app.post("/predict")
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X = preprocess_data(data)
        pred = model.predict(X)[0]

        prob = (
            model.predict_proba(X).max()
            if hasattr(model, "predict_proba")
            else None
        )

        return {
            "prediction": int(pred),
            "prediction_label": "Survived" if pred == 1 else "Did not survive",
            "probability": round(float(prob), 4) if prob else None
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/predict")
def predict_get(
    Pclass: int,
    Sex: str,
    Age: Optional[float] = None,
    SibSp: int = 0,
    Parch: int = 0,
    Fare: float = 0.0,
    Cabin: Optional[str] = None,
    Embarked: Optional[str] = None,
):
    input_data = InputData(
        Pclass=Pclass,
        Sex=Sex,
        Age=Age,
        SibSp=SibSp,
        Parch=Parch,
        Fare=Fare,
        Cabin=Cabin,
        Embarked=Embarked,
    )
    return predict(input_data)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
