import os
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse

# 1. FORCE ABSOLUTE PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_PATH = os.path.join(BASE_DIR, 'future_predictions.csv')
MODELS_REGISTRY_PATH = os.path.join(BASE_DIR, 'best_models_registry.pkl')

cache = {"predictions": None, "registry": None}

# 2. LOAD DATA WITH DIAGNOSTIC LOGS
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- STARTING DATA LOAD ---")
    print(f"Looking for CSV at: {PREDICTIONS_PATH}")
    
    try:
        if os.path.exists(PREDICTIONS_PATH):
            cache["predictions"] = pd.read_csv(PREDICTIONS_PATH, index_col=0)
            print("SUCCESS: future_predictions.csv loaded!")
        else:
            print("ERROR: CSV File not found at the path above!")

        if os.path.exists(MODELS_REGISTRY_PATH):
            cache["registry"] = joblib.load(MODELS_REGISTRY_PATH)
            print("SUCCESS: best_models_registry.pkl loaded!")
        else:
            print("ERROR: PKL File not found!")
            
    except Exception as e:
        print(f"CRITICAL ERROR during file load: {e}")

    yield
    cache.clear()

# 3. INITIALIZE APP
app = FastAPI(
    title="Sales Forecasting API",
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

# 4. NOW define your routes
@app.get("/")
async def read_index():
    return FileResponse('index.html')

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
        raise HTTPException(status_code=404, detail=f"State '{state}' not found.")
    
    forecast_data = predictions_df[matched_state].to_dict()
    model_used = cache["registry"].get(matched_state, "Unknown") if cache["registry"] else "Unknown"
    
    return {
        "state": matched_state,
        "best_model_algorithm_selected": model_used,
        "forecast_horizon_weeks": len(forecast_data),
        "predictions": forecast_data
    }
