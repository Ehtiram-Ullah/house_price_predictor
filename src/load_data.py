import pandas as pd

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    

df = load_data("src\\data\\Housing.csv")

print(df.head())