import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from src.train import PowerForecastModel

app   = FastAPI(title='Helios Power Forecast API')
model = PowerForecastModel.load('models/model.pkl')

class Row(BaseModel):
    timestamp:        str
    forecast_zephyr:  float
    forecast_boreas:  float
    feature_0:        float
    feature_1:        float
    feature_2:        float
    feature_3:        float
    feature_4:        float
    feature_5:        float
    feature_6:        float
    feature_7:        float
    feature_8:        float
    feature_9:        float
    feature_10:       float
    feature_11:       float
    feature_12:       float
    feature_13:       float
    feature_14:       float
    feature_15:       float
    feature_16:       float
    feature_17:       float
    feature_18:       float
    feature_19:       float

class PredictRequest(BaseModel):
    rows: List[Row]

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/predict')
def predict(request: PredictRequest):
    try:
        df = pd.DataFrame([r.model_dump() for r in request.rows])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        predictions = model.predict(df)
        return {'predictions': predictions}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
