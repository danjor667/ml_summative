import pickle
import numpy as np
import pandas as pd


with open("new_model.pkl", "rb") as file:
    MODEL = pickle.load(file)

with open("scaler1.pkl", "rb") as file2:
    SCALER = pickle.load(file2)




X = [
    [9,132,84,40,105,37.2,0.201,90]
]

features_names = ["Pregnancies",
                  "Glucose",
                  "BloodPressure",
                  "SkinThickness",
                  "Insulin",
                  "BMI",
                  "DiabetesPedigreeFunction",
                  "Age"
                  ]

X_new = np.array(X)


df = pd.DataFrame(X_new, columns=features_names)
df_scaled = SCALER.transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=features_names)


prob = MODEL.decision_function(df_scaled)

prediction  = MODEL.predict(df_scaled)


print(prob)

print(prediction)


