# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("arashnic/uplift-modeling")

# print("Path to dataset files:", path)

import pandas as pd
df=pd.read_csv('/root/test01/research/CausalCVR/dataset/1/criteo-uplift-v2.1.csv')
print(df.shape)
print(df.head)
df10 = df.sample(frac=0.1, replace=True, random_state=42)
print(df10.shape)
print(df10.head)
df10.to_pickle("data10.pkl")