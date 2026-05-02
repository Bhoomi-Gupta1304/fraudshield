"""
post.py  —  FraudShield ML helpers
────────────────────────────────────
Handles model loading, selection, and prediction.

IMPORTANT — COLUMN ORDER:
  train_models.py trains all models in frontend order: [V1..V28, Amount, Time]
  So features sent from fraudshield.html arrive in the same order the models
  expect. NO reordering is needed here.

  If you retrained with the old Kaggle order [Time, V1..V28, Amount],
  uncomment reorder_for_model() below and call it inside predict_transaction.
"""

import numpy as np
import joblib
import os


# ════════════════════════════════════════════════════════════════════════
#  COLUMN ORDER NOTE
#
#  Frontend sends  : [V1, V2, ..., V28, Amount, Time]   ← indices 0..29
#  Models trained  : same order (train_models.py fixes this)
#
#  If your models were trained on the RAW Kaggle CSV order:
#    [Time, V1..V28, Amount]
#  then uncomment and call reorder_for_model() inside predict_transaction.
# ════════════════════════════════════════════════════════════════════════

# def reorder_for_model(features):
#     """Only needed if models were trained with Time as first column."""
#     features = np.array(features, dtype=float)
#     single   = features.ndim == 1
#     if single: features = features.reshape(1, -1)
#     time_col   = features[:, 29:30]
#     v_cols     = features[:, 0:28]
#     amount_col = features[:, 28:29]
#     reordered  = np.hstack([time_col, v_cols, amount_col])
#     return reordered[0] if single else reordered


# ════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ════════════════════════════════════════════════════════════════════════

def load_models():
    """
    Load all three models + LR scaler from BASE_DIR.

    Returns
    -------
    models : dict  { "Logistic Regression": model|None,
                     "Random Forest":       model|None,
                     "XGBoost":             model|None }
    scaler : StandardScaler | None
    """
    base = os.path.dirname(os.path.abspath(__file__))

    model_files = {
        "Logistic Regression": "lr_model.pkl",
        "Random Forest":       "rf_model.pkl",
        "XGBoost":             "xgb_model.pkl",
    }

    models = {}
    for name, fname in model_files.items():
        path = os.path.join(base, fname)
        try:
            models[name] = joblib.load(path)
            print(f"  [OK]   Loaded {name}")
        except FileNotFoundError:
            models[name] = None
            print(f"  [--]   {fname} not found — {name} will use dummy fallback")
        except Exception as e:
            models[name] = None
            print(f"  [ERR]  {fname}: {e}")

    scaler = None
    try:
        scaler = joblib.load(os.path.join(base, "lr_scaler.pkl"))
        print("  [OK]   Loaded lr_scaler.pkl")
    except FileNotFoundError:
        print("  [--]   lr_scaler.pkl not found — LR will use unscaled features")
    except Exception as e:
        print(f"  [ERR]  lr_scaler.pkl: {e}")

    return models, scaler


# ════════════════════════════════════════════════════════════════════════
#  MODEL AUTO-SELECTION  (when frontend doesn't specify a model)
# ════════════════════════════════════════════════════════════════════════

def select_model_by_count(models: dict, num_transactions: int):
    """
    Automatically pick model based on transaction volume seen so far.

    < 10    → Logistic Regression  (fast, simple)
    < 100   → Random Forest        (balanced)
    >= 100  → XGBoost              (best accuracy)

    Returns (model_object, model_name_string)
    """
    if num_transactions < 10:
        name = "Logistic Regression"
    elif num_transactions < 100:
        name = "Random Forest"
    else:
        name = "XGBoost"

    return models[name], name


# ════════════════════════════════════════════════════════════════════════
#  SINGLE PREDICTION
# ════════════════════════════════════════════════════════════════════════

def predict_transaction(model,
                        features: np.ndarray,
                        model_name: str = "",
                        scaler=None):
    """
    Run inference on one transaction (shape 1×30).

    Parameters
    ----------
    model      : trained sklearn / xgboost model, or None (dummy fallback)
    features   : numpy array (1, 30) in order [V1..V28, Amount, Time]
    model_name : "Logistic Regression" | "Random Forest" | "XGBoost"
    scaler     : StandardScaler for LR only; pass None for RF / XGB

    Returns
    -------
    prediction  : int   1 = Fraud, 0 = Not Fraud
    probability : float fraud probability in [0, 1]
    """
    features = np.array(features, dtype=float).reshape(1, -1)

    if model is not None:
        # Scale input for Logistic Regression only
        if model_name == "Logistic Regression" and scaler is not None:
            X_in = scaler.transform(features)
        else:
            X_in = features

        probability = float(model.predict_proba(X_in)[0][1])

        # Use tuned threshold for LR (stored by train_models.py)
        # Default 0.5 causes LR to over-predict fraud after SMOTE training
        threshold = getattr(model, "best_threshold_", 0.5)
        prediction = 1 if probability >= threshold else 0

    else:
        # ── Dummy fallback (no model file found) ─────────────────────
        # beta(1,6) skews low → ~15% fraud rate, similar to real dataset
        probability = float(np.random.beta(a=1, b=6))
        prediction  = 1 if probability > 0.55 else 0

    return prediction, probability


# ════════════════════════════════════════════════════════════════════════
#  BATCH PREDICTION  (CSV upload)
# ════════════════════════════════════════════════════════════════════════

def predict_batch(model,
                  features_matrix: np.ndarray,
                  model_name: str = "XGBoost",
                  scaler=None):
    """
    Run inference on N transactions from a CSV upload.

    Returns a list of dicts that app.py returns as JSON:
    [
      { "row": 1, "prediction": "Fraud", "probability": 0.87,
        "risk": "High", "model": "XGBoost" },
      ...
    ]
    """
    features_matrix = np.array(features_matrix, dtype=float)
    results = []

    for idx in range(len(features_matrix)):
        row = features_matrix[idx].reshape(1, -1)
        pred, prob = predict_transaction(
            model, row,
            model_name=model_name,
            scaler=scaler,
        )
        results.append({
            "row":         idx + 1,
            "prediction":  "Fraud" if pred == 1 else "Not Fraud",
            "probability": round(prob, 4),
            "risk":        risk_label(prob),
            "model":       model_name,
        })

    return results


# ════════════════════════════════════════════════════════════════════════
#  RISK LABEL HELPER
# ════════════════════════════════════════════════════════════════════════

def risk_label(probability: float) -> str:
    """Map raw probability to human-readable risk tier."""
    if probability > 0.7:
        return "High"
    elif probability > 0.4:
        return "Medium"
    else:
        return "Low"