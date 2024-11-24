import pandas as pd
import uvicorn
from fastapi import FastAPI, Response
from pydantic import BaseModel
import pickle
import numpy as np


app = FastAPI()


with open("../ML/new_model.pkl", mode='rb') as f:
    model = pickle.load(file=f)

with open("../ML/scaler.pkl", mode='rb') as f1:
    scaler = pickle.load(file=f1)


class InputData(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


@app.post("/predict/")
def predict(data: InputData):

    variables = ["pregnancies"
               "glucose"
               "Bloodpressure"
               "SkinThickness"
               "Insulin"
               "BMI"
               "DiabetesPedigreeFunction"
               "Age"
               ]

    features  = [[data.Pregnancies, data.Glucose, data.BloodPressure, data.SkinThickness, data.Insulin, data.BMI,
                 data.DiabetesPedigreeFunction, data.Age]]

    ft = np.array(features)
    ft = pd.DataFrame(ft, columns=variables)

    scaled_ft = scaler.transform(ft)
    scaled_ft = pd.DataFrame(scaled_ft, columns=variables)

    try:
        prediction = model.predict(scaled_ft)
        probability = model.predict_proba(scaled_ft)
        return Response({"prediction": prediction, "probability": probability[0]}, status_code = 200 )
    except Exception as e:
        return {"error": str(e)}



if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8080)


