from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
from collections import deque
import numpy as np

app = Flask(__name__)
CORS(app)

# LOAD MODEL (trained using dataset)

model = joblib.load("cognitive_load_model.pkl")
scaler = joblib.load("scaler.pkl")
FEATURE_COLUMNS = joblib.load("feature_columns.pkl")

print(f"Loaded model with {len(FEATURE_COLUMNS)} features:")
print(FEATURE_COLUMNS)

# Buffer for smoothing predictions
prediction_buffer = deque(maxlen=5)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/mental")
def mental():
    return render_template("mental.html")

@app.route("/wellness")
def wellness():
    return render_template("wellness.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json


    # Basic input validation

    required_fields = [
        "speed", "avgInterval", "pause300", "pause500",
        "backspaceCount", "maxBurst", "editRatio", "duration"
    ]

    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    # Convert to float/int safely
    speed = float(data["speed"])
    avgInterval = float(data["avgInterval"])
    pause300 = int(data["pause300"])
    pause500 = int(data["pause500"])
    backspaceCount = int(data["backspaceCount"])
    maxBurst = int(data["maxBurst"])
    editRatio = float(data["editRatio"])
    duration = max(float(data["duration"]), 0.1)  # avoid zero division


    # Feature preparation (MATCH TRAINING EXACTLY)

    total_keys = max(int(speed * duration) + backspaceCount, 1)

    # Create feature dictionary matching EXACT order from training
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

    # Create DataFrame with features in the EXACT order the model expects
    features = pd.DataFrame([feature_dict])
    features = features[FEATURE_COLUMNS]  # Ensure correct order

    # Scale features
    features_scaled = scaler.transform(features)


    # Prediction + smoothing

    raw_score = float(model.predict(features_scaled)[0])
    prediction_buffer.append(raw_score)

    smoothed_score = np.mean(prediction_buffer)
    smoothed_score = float(np.clip(smoothed_score, 0, 100))


    # Cognitive load level

    if smoothed_score < 35:
        level = "low"
    elif smoothed_score < 65:
        level = "medium"
    else:
        level = "high"


    # Behavior-aware confidence

    error_penalty = min(backspaceCount * 2, 25)
    edit_penalty = min(editRatio * 0.8, 20)
    pause_penalty = min((pause300 + pause500) * 1.5, 20)

    stability_bonus = max(0, 15 - np.std(prediction_buffer) * 2) if len(prediction_buffer) > 1 else 0

    base_confidence = 85 - abs(smoothed_score - 50) * 0.6

    confidence = (
        base_confidence
        - error_penalty
        - edit_penalty
        - pause_penalty
        + stability_bonus
    )

    confidence = float(np.clip(confidence, 40, 95))


    # Response

    return jsonify({
        "score": round(smoothed_score, 2),
        "level": level,
        "confidence": round(confidence, 1)
    })

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting Cognitive Load Prediction API")
    print("="*50)
    print(f"Model features: {FEATURE_COLUMNS}")
    print("="*50 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=True)