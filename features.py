import pandas as pd
import numpy as np
import holidays

def create_features(df, state_name):
    df = df.copy()

    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month

    us_holidays = holidays.US(years=df.index.year.unique().tolist())
    df['IsHoliday'] = df.index.map(lambda d: int(d in us_holidays))

    df['Lag_1'] = df['Total'].shift(1)
    df['Lag_7'] = df['Total'].shift(7)
    df['Lag_30'] = df['Total'].shift(30)

    df['Rolling_Mean_7'] = df['Total'].rolling(window=7, min_periods=1).mean()
    df['Rolling_Std_7'] = df['Total'].rolling(window=7, min_periods=1).std()

    df['Rolling_Mean_30'] = df['Total'].rolling(window=30, min_periods=1).mean()
    df['Rolling_Std_30'] = df['Total'].rolling(window=30, min_periods=1).std()

    df = df.bfill()
    df['Rolling_Std_7'] = df['Rolling_Std_7'].fillna(0)
    df['Rolling_Std_30'] = df['Rolling_Std_30'].fillna(0)

    return df

if __name__ == "__main__":
    from data import prepare_data
    states_data = prepare_data('data.xlsx')
    df_feat = create_features(states_data['Alabama'], 'Alabama')
    print("Features for Alabama:")
    print(df_feat.head(10))
