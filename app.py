"""
app.py  —  FraudShield Flask Backend
──────────────────────────────────────
Run:  python app.py

On startup:
  • Checks if all 4 .pkl model files exist
  • If any is missing → auto-runs train_models.py first
  • Then loads models and starts Flask on port 5000

File layout expected:
  Credit_card/
  ├── app.py
  ├── post.py
  ├── train_models.py
  ├── fraudshield.html      ← served at http://127.0.0.1:5000/
  ├── creditcard.csv        ← needed for training only
  ├── templates/
  │   └── dashboard.html    ← served at http://127.0.0.1:5000/dashboard
  ├── lr_model.pkl          ← auto-created
  ├── lr_scaler.pkl         ← auto-created
  ├── rf_model.pkl          ← auto-created
  └── xgb_model.pkl         ← auto-created
"""

import os
import sys
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_FILES = ["lr_model.pkl", "rf_model.pkl", "xgb_model.pkl", "lr_scaler.pkl"]


# ════════════════════════════════════════════════════════════════════════
#  AUTO-TRAIN ON STARTUP
# ════════════════════════════════════════════════════════════════════════

def _models_exist() -> bool:
    return all(os.path.exists(os.path.join(BASE_DIR, f)) for f in MODEL_FILES)


def auto_train():
    print("\n" + "━"*60)
    print("  Model files missing — running train_models.py...")
    print("━"*60 + "\n")

    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

    try:
        import train_models
        success = train_models.run_training()
        if not success:
            print("\n  ⚠ Training failed (creditcard.csv missing?).")
            print("  App will start using dummy fallback predictions.\n")
    except ImportError:
        print("\n  ✗ train_models.py not found. Place it next to app.py.\n")
    except Exception as e:
        print(f"\n  ✗ Training error: {e}\n")
        print("  App will start using dummy fallback predictions.\n")


# Run before Flask even initialises
if not _models_exist():
    auto_train()
else:
    print("\n  ✓ All model files found — skipping training.")
    print("  Starting FraudShield server...\n")


# ════════════════════════════════════════════════════════════════════════
#  LOAD MODELS
# ════════════════════════════════════════════════════════════════════════
from post import (
    load_models,
    select_model_by_count,
    predict_transaction,
    predict_batch,
    risk_label,
)

print("\nLoading models...")
MODELS, LR_SCALER = load_models()
print("")

# Frontend shorthand → internal name
MODEL_ALIAS = {
    "logistic":            "Logistic Regression",
    "rf":                  "Random Forest",
    "xgboost":             "XGBoost",
    # full names also accepted
    "Logistic Regression": "Logistic Regression",
    "Random Forest":       "Random Forest",
    "XGBoost":             "XGBoost",
}

# Session counters (reset on server restart)
fraud_count      = 0
safe_count       = 0
last_probability = 0.0
model_usage      = {"Logistic Regression": 0, "Random Forest": 0, "XGBoost": 0}


# ════════════════════════════════════════════════════════════════════════
#  FLASK APP
# ════════════════════════════════════════════════════════════════════════
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
CORS(app)


def _resolve_model(data: dict):
    """
    Pick which model to use from the request body.
    Priority: explicit 'model' key > 'num_transactions' auto-select > XGBoost default
    """
    key = data.get("model", "")
    if key and key in MODEL_ALIAS:
        name = MODEL_ALIAS[key]
        return MODELS[name], name
    if "num_transactions" in data:
        return select_model_by_count(MODELS, int(data["num_transactions"]))
    return MODELS["XGBoost"], "XGBoost"


# ── GET /  →  fraudshield.html ────────────────────────────────────────
@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "fraudshield.html")


# ── GET /dashboard  →  templates/dashboard.html ───────────────────────
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# ════════════════════════════════════════════════════════════════════════
#  POST /predict
#  Body: { "features": [30 floats], "model": "xgboost"|"rf"|"logistic" }
#  Returns: { "prediction": "Fraud"|"Not Fraud",
#             "probability": 0.87, "model_used": "XGBoost" }
# ════════════════════════════════════════════════════════════════════════
@app.route("/predict", methods=["POST"])
def predict():
    global fraud_count, safe_count, last_probability, model_usage

    try:
        data     = request.get_json(force=True)
        features = np.array(data["features"], dtype=float).reshape(1, -1)

        if features.shape[1] != 30:
            return jsonify({"error": f"Expected 30 features, got {features.shape[1]}"}), 400

        model, model_name = _resolve_model(data)

        prediction, probability, reasons = predict_transaction(
            model, features,
            model_name=model_name,
            scaler=LR_SCALER if model_name == "Logistic Regression" else None,
        )

        label = "Fraud" if prediction == 1 else "Not Fraud"

        last_probability         = probability
        model_usage[model_name] += 1
        if label == "Fraud": fraud_count += 1
        else:                safe_count  += 1

        return jsonify({
            "prediction":     label,
            "probability":    round(probability, 4),
            "model_used":     model_name,
            "verdict_reason": reasons["verdict_reason"],
            "factors":        reasons["factors"],
            "feature_flags":  reasons["feature_flags"],
            "signal_count":   reasons["signal_count"],
            "summary":        reasons["summary"],
        })

    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ════════════════════════════════════════════════════════════════════════
#  POST /upload
#  Multipart form-data — field name "file", CSV with 30 columns.
#  Returns JSON array, one object per row.
# ════════════════════════════════════════════════════════════════════════
@app.route("/upload", methods=["POST"])
def upload_file():
    global fraud_count, safe_count

    try:
        file = request.files.get("file")
        if file is None:
            return jsonify({"error": "No file — field name must be 'file'"}), 400

        df = pd.read_csv(file)

        # Drop an accidental all-string header row
        if df.shape[0] > 0 and df.iloc[0].apply(lambda x: isinstance(x, str)).all():
            df = df.iloc[1:].reset_index(drop=True)

        df = df.apply(pd.to_numeric, errors="coerce").dropna()

        if df.shape[1] != 30:
            return jsonify({"error": f"CSV must have 30 columns, got {df.shape[1]}"}), 400

        model, model_name = select_model_by_count(MODELS, len(df))

        results = predict_batch(
            model, df.values,
            model_name=model_name,
            scaler=LR_SCALER if model_name == "Logistic Regression" else None,
        )

        for r in results:
            if r["prediction"] == "Fraud": fraud_count += 1
            else:                          safe_count  += 1

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ════════════════════════════════════════════════════════════════════════
#  GET /stats  —  polled by dashboard every 3 s
# ════════════════════════════════════════════════════════════════════════
@app.route("/stats")
def stats():
    return jsonify({
        "fraud":       fraud_count,
        "safe":        safe_count,
        "total":       fraud_count + safe_count,
        "probability": last_probability,
        "models":      model_usage,
    })


# ════════════════════════════════════════════════════════════════════════
#  GET /simulate  —  convenience endpoint; Live Feed uses /predict
# ════════════════════════════════════════════════════════════════════════
@app.route("/simulate")
def simulate():
    global fraud_count, safe_count, last_probability

    fake = np.random.randn(1, 30).astype(float)
    model, model_name = select_model_by_count(MODELS, 50)   # always RF

    prediction, probability, reasons = predict_transaction(
        model, fake, model_name=model_name, scaler=None
    )
    label = "Fraud" if prediction == 1 else "Not Fraud"

    last_probability         = probability
    model_usage[model_name] += 1
    if label == "Fraud": fraud_count += 1
    else:                safe_count  += 1

    return jsonify({
        "prediction":     label,
        "probability":    round(probability, 4),
        "model_used":     model_name,
        "verdict_reason": reasons["verdict_reason"],
        "factors":        reasons["factors"],
        "feature_flags":  reasons["feature_flags"],
        "signal_count":   reasons["signal_count"],
    })


# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=True, port=5000)