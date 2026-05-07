import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Loading data...")
    df = pd.read_excel('data.xlsx')

    print("\n--- Basic Info ---")
    print(f"Shape: {df.shape}")
    print(df.dtypes)

    print("\n--- Missing Values ---")
    print(df.isna().sum())

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['State', 'Date']).reset_index(drop=True)

    print("\n--- Date Frequency across States ---")
    freq = df.groupby('State')['Date'].diff().value_counts()
    print(freq)

    print("\n--- Records per State ---")
    state_counts = df.groupby('State')['Date'].count()
    print(state_counts.head())
    print(f"Total States: {len(state_counts)}")

if __name__ == "__main__":
    main()
