# backend/routers/predict.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

router = APIRouter()



MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "backend/models", "model.pkl")

model = joblib.load(MODEL_PATH)
# Load once at module start

class PowerInput(BaseModel):
    AT: float  # Ambient Temperature (°C)
    V: float   # Exhaust Gas Velocity (m/s)
    AP: float  # Atmospheric Pressure (mbar)
    RH: float  # Relative Humidity (%)

@router.post("/")
async def predict_power(data: PowerInput):
    try:
        input_df = pd.DataFrame([data.dict()])
        predicted_pe = model.predict(input_df)[0]
        return {
            "Predicted_Power_Output": round(predicted_pe, 2),
            "Units": "Watts",
            "Input_Conditions": {
                "Ambient_Temperature": f"{data.AT} °C",
                "Velocity": f"{data.V} m/s",
                "Pressure": f"{data.AP} mbar",
                "Humidity": f"{data.RH} %"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
