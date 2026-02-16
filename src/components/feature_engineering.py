import pandas as pd
from sklearn.preprocessing import StandardScaler

class FeatureEngineering:

    def __init__(self):
        self.geo_scaler = StandardScaler()

    # --------------------
    # Temporal Features
    # --------------------

    def extract_temporal_features(self,df:pd.DataFrame) -> pd.DataFrame:

        df['Hour'] = df['Date'].dt.hour
        df['DayOfWeek_Num'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Is_Weekend'] = df['DayOfWeek_Num'].isin([5,6]).astype(int)

        df['Season'] = df['Month'].map({

            12:'Winter', 1:'Winter', 2:'Winter',
            3:'Spring', 4:'Spring', 5:'Spring',
            6:'Summer', 7:'Summer', 8:'Summer',
            9:'Fall', 10: 'Fall', 11:'Fall'
        })
        return df
    # -----------------------
    # Crime Severity
    # -----------------------

    def add_crime_severity(self,df:pd.DataFrame) -> pd.DataFrame:
        severity_map = {
            # Level 4 – Extreme
            'HOMICIDE': 4,
            'CRIM SEXUAL ASSAULT': 4,
            'KIDNAPPING': 4,
            'OFFENSE INVOLVING CHILDREN': 4,

            # Level 3 – Serious violent / weapons
            'ROBBERY': 3,
            'AGGRAVATED ASSAULT': 3,
            'WEAPONS VIOLATION': 3,
            'SEX OFFENSE': 3,

            # Level 2 – Property / drugs / economic
            'THEFT': 2,
            'BURGLARY': 2,
            'MOTOR VEHICLE THEFT': 2,
            'NARCOTICS': 2,
            'DECEPTIVE PRACTICE': 2,

            # Level 1 – Minor / public order
            'CRIMINAL TRESPASS': 1,
            'PUBLIC PEACE VIOLATION': 1,
            'LIQUOR LAW VIOLATION': 1,
            'GAMBLING': 1,
            'OBSCENITY': 1
        }
        df['Crime_Severity_Score'] = (
            df['Primary Type'].map(severity_map).fillna(2)
        )
        return df
    
    # ----------------------
    # Geographic Scaling
    # ----------------------
    def scale_geograph(self,df: pd.DataFrame) -> pd.DataFrame:
        geo_features = df[['Latitude','Longitude']]

        scaled = self.geo_scaler.fit_transform(geo_features)

        df['Lat_scaled'] = scaled[:,0]
        df['Long_scaled'] = scaled[:,1]

        return df
    
    # -----------------------
    # Run All
    # -----------------------
    def run(self,df:pd.DataFrame) -> pd.DataFrame:
        df = self.extract_temporal_features(df)
        df = self.add_crime_severity(df)
        df = self.scale_geograph(df)

        return df