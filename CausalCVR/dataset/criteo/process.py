import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("/root/test01/research/CausalCVR/dataset/criteo/data.csv")
train_df, test_df = train_test_split(df, test_size=0.33, random_state=42)
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)