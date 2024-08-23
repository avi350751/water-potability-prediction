import pandas as pd
import numpy as np
import os


#Fetch file from data/raw folder
train_data = pd.read_csv("./data/raw/train.csv")
test_data = pd.read_csv("./data/raw/test.csv")

#Preprocess the files
def fill_missing_value(df):
    #fill missing value with median
    for col in df.columns:
        if df[col].isnull().any():
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
    return df

train_processed = fill_missing_value(train_data)
test_processed = fill_missing_value(test_data)

#Save the data in data/processed folders
data_path = os.path.join("data", "processed")
os.makedirs(data_path)

train_processed.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
test_processed.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)