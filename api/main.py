import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle


app = FastAPI()


with open("../ML/new_model.pkl", mode='rb') as f:
    model = pickle.load(file=f)


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
    features  = [[data.Pregnancies, data.Glucose, data.BloodPressure, data.SkinThickness, data.Insulin, data.BMI,
                 data.DiabetesPedigreeFunction, data.Age]]
    try:
        prediction = model.predict(features)
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}



if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8080
    uvicorn.run(app, host=host, port=port)


