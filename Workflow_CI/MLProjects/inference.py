import joblib
import requests

def prediction(data):
    # URL endpoint dari model yang sedang di-serve
    url = "http://127.0.0.1:5000/invocations"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=data, headers=headers)
    response = response.json().get("predictions")
    result_target = joblib.load("model/encoder_target.joblib")
    final_result = result_target.inverse_transform(response)
    return final_result