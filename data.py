import pandas as pd
import numpy as np

def prepare_data(file_path):
    print("Loading data...")
    df = pd.read_excel(file_path)
    df['Date'] = pd.to_datetime(df['Date'])

    df = df.sort_values(by=['State', 'Date']).reset_index(drop=True)

    states_data = {}
    for state, group in df.groupby('State'):
        group = group.set_index('Date')

        weekly_group = group[['Total']].resample('W').sum()

        weekly_group['Total'] = weekly_group['Total'].replace(0, np.nan)

        weekly_group['Total'] = weekly_group['Total'].interpolate(method='linear')

        weekly_group['Total'] = weekly_group['Total'].bfill().ffill()

        states_data[state] = weekly_group

    print(f"Prepared regular weekly sequences for {len(states_data)} states.")
    print("Sample for Alabama (head):")
    print(states_data['Alabama'].head())
    print("\nTotal missing over all processed states:", sum(v.isna().sum().sum() for v in states_data.values()))

    return states_data

if __name__ == "__main__":
    prepare_data('data.xlsx')
