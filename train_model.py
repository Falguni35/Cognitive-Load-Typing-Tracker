import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)
# 1. LOAD DATASET

df = pd.read_csv("dataset.csv")
print("Dataset loaded:", df.shape[0], "samples")


# 2. FEATURE ENGINEERING (REDUCED)


df["total_keys"] = (df["speed"] * df["duration"]).astype(int) + df["backspaceCount"]
df["total_keys"] = df["total_keys"].replace(0, 1)

df["pause_ratio_300"] = df["pause300"] / df["total_keys"]
df["correction_rate"] = df["backspaceCount"] / df["total_keys"]


# 3. TARGET VARIABLE (WITH MORE NOISE)

label_to_score = {
    "low": 25,
    "medium": 55,
    "high": 75
}

df["cognitive_load_score"] = df["label"].map(label_to_score)
# Increased noise to make it more realistic
df["cognitive_load_score"] += np.random.normal(0, 8, size=len(df))


# 4. FEATURES (REDUCED SET)


FEATURE_COLUMNS = [
    "speed",
    "avgInterval",
    "pause300",
    "pause500",
    "backspaceCount",
    "maxBurst",
    "editRatio",
    "duration",
    "pause_ratio_300",
    "correction_rate"
    # Removed: pause_ratio_500, burst_instability
]

X = df[FEATURE_COLUMNS]
y = df["cognitive_load_score"]


# 5. TRAIN–TEST SPLIT (STRATIFIED)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=100,
    stratify=df["label"]  # Ensures balanced classes
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. MODEL TRAINING (REGULARIZED)
model = GradientBoostingRegressor(
    n_estimators=50,
    learning_rate=0.15,
    max_depth=3,
    min_samples_split=15,
    min_samples_leaf=8,
    subsample=0.7,
    max_features='sqrt',
    random_state=42
)

model.fit(X_train_scaled, y_train)

# 7. REGRESSION METRICS

preds = model.predict(X_test_scaled)
print("\nRegression Metrics:")
print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
print(f"R2 Score: {r2_score(y_test, preds):.3f}")


# 8. REGRESSION → CLASS METRICS


def score_to_label(score):
    if score < 35:
        return "low"
    elif score < 65:
        return "medium"
    else:
        return "high"

y_test_labels = y_test.apply(score_to_label)
y_pred_labels = pd.Series(preds).apply(score_to_label)

accuracy = accuracy_score(y_test_labels, y_pred_labels)
f1 = f1_score(y_test_labels, y_pred_labels, average="weighted")
cm = confusion_matrix(
    y_test_labels,
    y_pred_labels,
    labels=["low", "medium", "high"]
)

print("\nClassification Metrics:")
print(f"Accuracy: {accuracy:.3f}")
print(f"F1 Score: {f1:.3f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred_labels))


# 9. CROSS-VALIDATION


kf = KFold(n_splits=5, shuffle=True, random_state=42)

X_scaled_full = scaler.fit_transform(X)

cv_scores = cross_val_score(
    model,
    X_scaled_full,
    y,
    cv=kf,
    scoring="r2"
)

print("\nCross-Validation (5-Fold R2):")
print(f"Mean: {cv_scores.mean():.3f}")
print(f"Std Dev: {cv_scores.std():.3f}")

# 10. SAVE MODEL

joblib.dump(model, "cognitive_load_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(FEATURE_COLUMNS, "feature_columns.pkl")

print("\n✅ Model, scaler & feature list saved successfully!")