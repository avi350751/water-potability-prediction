import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

#Fetch the data
df = pd.read_csv("/Users/avi/Desktop/CampusX/datasets/water_potability_01.csv")

#split the data
train_df, test_df = train_test_split(df, test_size=0.2, random_state= 42)

#create the folders
data_path = os.path.join("data", "raw")
os.makedirs(data_path)

#Save train and test data
train_df.to_csv(os.path.join(data_path, "train.csv"), index=False)
test_df.to_csv(os.path.join(data_path, "test.csv"), index=False)
