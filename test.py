import pandas as pd

df = pd.read_csv("artifacts/processed_dataset.csv")

# Take smaller subset for cloud
df_small = df.sample(n=50000, random_state=42)

df_small.to_csv("artifacts/deployment_dataset.csv", index=False)
