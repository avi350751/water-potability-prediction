import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

test_df = pd.read_csv("./data/processed/test_processed.csv")

X_test = test_df.iloc[:,0:-1].values
y_test = test_df.iloc[:,-1].values

model = pickle.load(open('model.pkl', 'rb'))

y_pred = model.predict(X_test)

#Metrics
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

metrics_dict = {

    'accuracy' : acc,
    'precision' : precision,
    'recall': rec,
    'f1_score' : f1
}

with open('metrics.json', 'w') as file:
    json.dump(metrics_dict, file, indent = 4)