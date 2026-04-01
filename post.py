import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "features": [0]*30   # dummy input (must match number of features)
}

response = requests.post(url, json=data)

print(response.json())