from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predict import load_artifacts, predict_class, predict_proba


app = FastAPI(title="Wine Model API (RandomForest)", version="1.0")


class PredictRequest(BaseModel):
    features: List[float] = Field(..., description="List of 13 numeric feature values in Wine dataset order.")


class PredictResponse(BaseModel):
    prediction: int
    prediction_label: str


class ProbaResponse(BaseModel):
    probabilities: List[float]
    classes: List[str]


@app.on_event("startup")
def startup_check():
    load_artifacts()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    _, metadata = load_artifacts()
    return metadata


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        result = predict_class(req.features)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_proba", response_model=ProbaResponse)
def predict_probability(req: PredictRequest):
    try:
        result = predict_proba(req.features)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))