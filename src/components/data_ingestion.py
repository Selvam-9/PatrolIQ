import os
import pandas as pd


class DataIngestion:

    def __init__(self, raw_path: str, processed_path: str):
        self.raw_path = raw_path
        self.processed_path = processed_path

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.raw_path)
        return df

    def sample_recent_data(self, df: pd.DataFrame, n_samples: int = 500000) -> pd.DataFrame:
        df['Date'] = pd.to_datetime(
            df['Date'],
            errors='coerce'
        )
        df = df.sort_values(by='Date', ascending=False)
        sampled_df = df.head(n_samples)

        return sampled_df


    def save_processed_data(self, df: pd.DataFrame):
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        df.to_csv(self.processed_path, index=False)

    def run(self):
        df = self.load_data()
        sampled_df = self.sample_recent_data(df)
        self.save_processed_data(sampled_df)
        return sampled_df