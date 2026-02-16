import pandas as pd


class DataPreprocessing:

    def __init__(self):
        pass

    def drop_missing_geo(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=['Latitude', 'Longitude'])
        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates(subset=['ID'])
        return df

    def enforce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = [
            'Latitude',
            'Longitude',
            'Beat',
            'District',
            'Ward',
            'Community Area',
            'Year'
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.drop_missing_geo(df)
        df = self.remove_duplicates(df)
        df = self.enforce_types(df)
        return df
