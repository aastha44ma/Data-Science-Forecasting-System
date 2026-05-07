import pandas as pd
import numpy as np
from data import prepare_data
from features import create_features
from models import train_sarima, train_prophet, train_xgboost, train_lstm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

def evaluate_models_for_state(state_name, df):
    val_len = 8
    train = df.iloc[:-val_len]
    val = df.iloc[-val_len:]

    results = {}
    print(f"\n--- Training Models for {state_name} ---")

    print("Training SARIMA...")
    sarima_model, sarima_preds = train_sarima(train['Total'], val_len)
    results['SARIMA'] = mean_absolute_error(val['Total'], sarima_preds)

    print("Training Prophet...")
    prophet_model, prophet_preds = train_prophet(train, val_len)
    results['Prophet'] = mean_absolute_error(val['Total'], prophet_preds)

    print("Training XGBoost...")
    xgb_model, xgb_preds = train_xgboost(train, val, target_col='Total')
    results['XGBoost'] = mean_absolute_error(val['Total'], xgb_preds)

    print("Training LSTM...")
    try:
        lstm_model, lstm_preds = train_lstm(train, val, target_col='Total')
        results['LSTM'] = mean_absolute_error(val['Total'], lstm_preds)
    except Exception as e:
        print("LSTM Failed", e)
        results['LSTM'] = np.inf

    print(f"[{state_name}] MAE Results:\n", results)
    best_model = min(results, key=results.get)
    print(f"🏆 Best Model for {state_name}: {best_model} (MAE: {results[best_model]:.2f})")
    return best_model, results

def main():
    states_data = prepare_data('data.xlsx')

    best_models = {}
    mae_results = {}

    test_states = ['California', 'Texas', 'New York', 'Florida', 'Illinois']

    for state in test_states:
        if state not in states_data:
            print(f"Skipping {state}, not in dataset.")
            continue

        df = states_data[state]
        df_feat = create_features(df, state)

        best, maes = evaluate_models_for_state(state, df_feat)
        best_models[state] = best
        mae_results[state] = maes

    print("\n========= Summary =========")
    for state in best_models:
        print(f"{state}: Best -> {best_models[state]}")

if __name__ == "__main__":
    main()
