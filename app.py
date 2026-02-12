from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
from collections import deque
import numpy as np
import os
# APP SETUP

app = Flask(__name__)
CORS(app)

# LOAD MODEL SAFELY (RENDER SAFE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "cognitive_load_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
FEATURE_COLUMNS = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))

prediction_buffer = deque(maxlen=5)

# TEMPLATE ROUTES

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/mental")
def mental():
    return render_template("mental.html")

@app.route("/wellness")
def wellness():
    return render_template("wellness.html")

# PREDICTION API

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    required_fields = [
        "speed", "avgInterval", "pause300", "pause500",
        "backspaceCount", "maxBurst", "editRatio", "duration"
    ]

    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    # Convert inputs safely
    speed = float(data["speed"])
    avgInterval = float(data["avgInterval"])
    pause300 = int(data["pause300"])
    pause500 = int(data["pause500"])
    backspaceCount = int(data["backspaceCount"])
    maxBurst = int(data["maxBurst"])
    editRatio = float(data["editRatio"])
    duration = max(float(data["duration"]), 0.1)

    # Feature engineering (MATCH TRAINING)
    total_keys = max(int(speed * duration) + backspaceCount, 1)

    feature_dict = {
        "speed": speed,
        "avgInterval": avgInterval,
        "pause300": pause300,
        "pause500": pause500,
        "backspaceCount": backspaceCount,
        "maxBurst": maxBurst,
        "editRatio": editRatio,
        "duration": duration,
        "pause_ratio_300": pause300 / total_keys,
        "correction_rate": backspaceCount / total_keys
    }

    features = pd.DataFrame([feature_dict])[FEATURE_COLUMNS]
    features_scaled = scaler.transform(features)

    # Prediction + smoothing
    raw_score = float(model.predict(features_scaled)[0])
    prediction_buffer.append(raw_score)

    smoothed_score = float(np.clip(np.mean(prediction_buffer), 0, 100))

    # Load level
    if smoothed_score < 35:
        level = "low"
    elif smoothed_score < 65:
        level = "medium"
    else:
        level = "high"

    # Confidence calculation
    error_penalty = min(backspaceCount * 2, 25)
    edit_penalty = min(editRatio * 0.8, 20)
    pause_penalty = min((pause300 + pause500) * 1.5, 20)

    stability_bonus = (
        max(0, 15 - np.std(prediction_buffer) * 2)
        if len(prediction_buffer) > 1
        else 0
    )

    base_confidence = 85 - abs(smoothed_score - 50) * 0.6

    confidence = base_confidence - error_penalty - edit_penalty - pause_penalty + stability_bonus
    confidence = float(np.clip(confidence, 40, 95))

    return jsonify({
        "score": round(smoothed_score, 2),
        "level": level,
        "confidence": round(confidence, 1)
    })


# LOCAL RUN (IGNORED BY RENDER)

if __name__ == "__main__":
    app.run(debug=True)
