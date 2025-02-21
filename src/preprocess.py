import pandas as pd

def load_data(file_path="data/sentiment_data.csv"):
    """Loads the dataset into a Pandas DataFrame."""
    df = pd.read_csv(file_path)
    print(df.head())  # Check the first few rows
    return df

if __name__ == "__main__":
    data = load_data()
