import pandas as pd

df = pd.read_csv("C:/Users/hp/Desktop/advanced/Capstone-Project/data/processed/SWaT_cleaned.csv")

attack_row = df[df["Normal/Attack"] == 1].iloc[0]

features = attack_row.drop(
    labels=["Timestamp", "Normal/Attack"]
).tolist()

print("Number of features:", len(features))
print(features)
