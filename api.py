from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse

@app.get("/")
async def read_index():
    return FileResponse('index.html')

cache = {
    "predictions": None,
    "registry": None
}

PREDICTIONS_PATH = 'future_predictions.csv'
MODELS_REGISTRY_PATH = 'best_models_registry.pkl'

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(PREDICTIONS_PATH):
        cache["predictions"] = pd.read_csv(PREDICTIONS_PATH, index_col=0)
    if os.path.exists(MODELS_REGISTRY_PATH):
        cache["registry"] = joblib.load(MODELS_REGISTRY_PATH)
    yield
    cache.clear()

app = FastAPI(
    title="Sales Forecasting API", 
    description="Ultra-lightweight backend service exposing 8-Week State Sales Forecasts.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "status": "Online",
        "message": "Welcome to the State Sales Forecasting API.",
        "usage": "Send a GET request to /forecast/{state_name} to view projections."
    }

@app.get("/states")
def get_supported_states():
    if cache["predictions"] is None:
        raise HTTPException(status_code=500, detail="Prediction data not found.")
    return {"states": cache["predictions"].columns.tolist()}

@app.get("/forecast/{state}")
def get_state_forecast(state: str):
    if cache["predictions"] is None:
        raise HTTPException(status_code=500, detail="Server hasn't loaded prediction data properly.")

    predictions_df = cache["predictions"]
    states = predictions_df.columns.tolist()

    matched_state = next((s for s in states if s.lower() == state.lower()), None)

    if not matched_state:
        raise HTTPException(
            status_code=404, 
            detail=f"State '{state}' not found. Verify the state name by calling the /states directory."
        )

    forecast_data = predictions_df[matched_state].to_dict()
    model_used = cache["registry"].get(matched_state, "Unknown") if cache["registry"] else "Unknown"

    return {
        "state": matched_state,
        "best_model_algorithm_selected": model_used,
        "forecast_horizon_weeks": len(forecast_data),
        "predictions": forecast_data
    }
