import requests



metrics = {"pregnancies": 3,
            "glucose": 148,
            "Bloodpressure": 72,
            "SkinThickness": 35,
            "Insulin": 94,
            "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Age": 25,
            }

def make_prediction(data):
    url = 'http://127.0.0.1:8080/predict/'

    response = requests.post(url, json=data)

    return response


prediction = make_prediction(metrics)

print(prediction.json())