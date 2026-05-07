import pandas as pd
import numpy as np
import warnings
import joblib
from data import prepare_data
from features import create_features
from models import train_sarima, train_prophet, train_xgboost, train_lstm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

warnings.filterwarnings('ignore')

def train_and_select_best_model(df_feat, state_name, val_len=8):
    train = df_feat.iloc[:-val_len]
    val = df_feat.iloc[-val_len:]

    results = {}
    print(f"\nEvaluating models for {state_name}...")

    try:
        _, p_sarima = train_sarima(train['Total'], val_len)
        results['SARIMA'] = mean_absolute_error(val['Total'], p_sarima)
    except:
        results['SARIMA'] = np.inf

    try:
        _, p_prophet = train_prophet(train, val_len)
        results['Prophet'] = mean_absolute_error(val['Total'], p_prophet)
    except:
        results['Prophet'] = np.inf

    try:
        _, p_xgb = train_xgboost(train, val, target_col='Total')
        results['XGBoost'] = mean_absolute_error(val['Total'], p_xgb)
    except:
        results['XGBoost'] = np.inf

    try:
        _, p_lstm = train_lstm(train, val, target_col='Total')
        results['LSTM'] = mean_absolute_error(val['Total'], p_lstm)
    except:
        results['LSTM'] = np.inf

    best_model = min(results, key=results.get)
    print(f"[{state_name}] Best Model: {best_model} | MAEs: {results}")
    return best_model, results

def generate_future_features(df_feat, steps=8):
    """
    Generate the next `steps` future rows for forecasting.
    This reconstructs the lag/rolling features iteratively or roughly.
    For simplicity in this case study, we just append dates and recreate features context.
    """
    last_date = df_feat.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=steps, freq='W')
    future_df = pd.DataFrame(index=future_dates, columns=['Total'])

    combined = pd.concat([df_feat[['Total']], future_df])

    return combined

def forecast_future(df_feat, best_model_type, steps=8):
    """
    Train best model on FULL historical data and forecast the next 8 weeks.
    """
    forecasts = []

    train = df_feat.copy()

    if best_model_type == 'SARIMA':
        fitted, forecast = train_sarima(train['Total'], steps)
        forecasts = forecast.values

    elif best_model_type == 'Prophet':
        m, forecast = train_prophet(train, steps)
        forecasts = forecast

    elif best_model_type == 'XGBoost':
        combined = generate_future_features(df_feat, steps)

        temp_df = train.copy()

        for i in range(steps):
            comb = generate_future_features(temp_df, 1)
            comb_feat = create_features(comb, 'temp')

            comb_feat = comb_feat.astype(float)

            X_train = comb_feat.iloc[:-1].drop(columns=['Total'])
            y_train = comb_feat.iloc[:-1]['Total']
            X_test = comb_feat.iloc[[-1]].drop(columns=['Total'])

            import xgboost as xgb
            model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)[0]

            next_date = comb_feat.index[-1]
            temp_df.loc[next_date, 'Total'] = pred
            forecasts.append(pred)

    elif best_model_type == 'LSTM':
        val_dummy = train.iloc[-1:] 
        model, _ = train_lstm(train, val_dummy, target_col='Total')

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        train_data = train.values
        scaler.fit(train_data)
        train_scaled = scaler.transform(train_data)

        import torch
        seq_length = 8
        current_seq = torch.tensor(train_scaled[-seq_length:], dtype=torch.float32).unsqueeze(0)

        preds = []
        model.eval()

        temp_df = train.copy()

        for i in range(steps):
             with torch.no_grad():
                pred_scaled = model(current_seq).item()
                preds.append(pred_scaled)

                comb = generate_future_features(temp_df, 1)

                next_step = train_scaled[-1].copy()
                next_step[0] = pred_scaled
                next_step_t = torch.tensor(next_step, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                current_seq = torch.cat([current_seq[:, 1:, :], next_step_t], dim=1)

        dummy = np.zeros((len(preds), train_scaled.shape[1]))
        dummy[:, 0] = preds
        forecasts = scaler.inverse_transform(dummy)[:, 0]

    return forecasts

def main():
    print("Initializing Forecasting System...")
    states_data = prepare_data('data.xlsx')

    registry = {}
    future_predictions = {}

    for state, df in states_data.items():
        df_feat = create_features(df, state)

        best_model, maes = train_and_select_best_model(df_feat, state)
        registry[state] = best_model

        print(f"[{state}] Forecasting next 8 weeks using {best_model}...")
        future_forecast = forecast_future(df_feat, best_model, steps=8)
        future_predictions[state] = future_forecast

        print(f"[{state}] Forecast: {np.round(future_forecast, 2)}")
        print("-" * 50)

    print("\nTraining Complete.")
    print("Saving Models Registry...")
    joblib.dump(registry, 'artifacts/best_models_registry.pkl')

    pred_df = pd.DataFrame(future_predictions)
    pred_df.index = [f"Week_{i+1}" for i in range(8)]
    pred_df.to_csv('artifacts/future_predictions.csv')
    print("Predictions saved to artifacts/future_predictions.csv")

if __name__ == "__main__":
    main()
