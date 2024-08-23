from fastapi import FastAPI
import pickle
import pandas as pd
from data_model import Water

app = FastAPI(

    title = "Water Potability",
    description = "Predicting Water Potability"
)

with open("/Users/avi/Desktop/CampusX/MLOPS/ml-pipeline/model.pkl", "rb") as file:
    model = pickle.load(file)

@app.get("/")
def index():
    return "Welcome to water potability prediction"

@app.post("/predict")
def model_predict(water : Water):
    sample = pd.DataFrame({
        'ph': [water.ph],
        'Hardness' : [water.Hardness],
        'Solids' : [water.Solids],
        'Chloramines' : [water.Chloramines],
        'Sulfate' : [water.Sulfate],
        'Conductivity' : [water.Conductivity],
        'Organic_carbon' : [water.Organic_carbon],
        'Trihalomethanes' : [water.Trihalomethanes],
        'Turbidity' : [water.Turbidity]
    })

    predicted_value = model.predict(sample)

    if predicted_value == 1:
        return 'water is consumable'
    else:
        return 'water is not consumable'