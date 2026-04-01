from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("fraud_model.pkl")

# Home route
@app.route("/")
def home():
    return "Fraud Detection API is running"

# 🔥 Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json   # get input data

    # Convert input to numpy array
    input_data = np.array(data["features"]).reshape(1, -1)

    # Get probability
    prob = model.predict_proba(input_data)[0][1]

    # Apply threshold (IMPORTANT)
    threshold = 0.2

    if prob > threshold:
        result = "Fraud"
    else:
        result = "Safe"

    return jsonify({
        "prediction": result,
        "probability": float(prob)
    })

if __name__ == "__main__":
    app.run(debug=True)