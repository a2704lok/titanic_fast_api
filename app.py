import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException # type: ignore
from pydantic import BaseModel
import numpy as np
codesetup_path = Path.home() / "Desktop" / "titanic_setup"
sys.path.append(str(codesetup_path))
from enumerate import Features
from predictor import XgboostPrediction
from exceptions import FileAbsenceError
import logging 


xgb_pred = XgboostPrediction(path=str(codesetup_path / "xgboost.pkl"))


logging.basicConfig( filename=str(codesetup_path)+"/app_logs.log", level=logging.INFO,
)


app = FastAPI()

@app.post("/survivor_pred")
async def predict_survivor_endpoint(features: Features):
    try:
        pred = xgb_pred.predict(features)
        return {"result":  pred.tolist()}
    
    except FileAbsenceError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
