# train_models.py
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# === 1. Paths ===
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic_oop_multiclass_realistic_10k.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# === 2. Topics and features ===
TOPICS = ["Class", "Constructor", "Encapsulation", "Inheritance", "Interface", "Polymorphism"]
FEATURE_COLS = [f"{t}%" for t in TOPICS]   # use the six percentage columns as input features

def level_from_score(x: float) -> int:
    """Convert numeric percentage into 0/1/2 = beginner/intermediate/advanced."""
    if x <= 30:
        return 0  # beginner
    elif x <= 70:
        return 1  # intermediate
    else:
        return 2  # advanced

for topic in TOPICS:
    print(f"\n=== Training model for topic: {topic} ===")
    score_col = f"{topic}%"

    # Target: bin the percentage into 0/1/2
    y = df[score_col].apply(level_from_score)

    # Features: all six topic percentages
    X = df[FEATURE_COLS]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, Y_train)

    # Optional: quick check
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred))

    # Save model
    model_path = os.path.join(MODELS_DIR, f"rf_model_{topic}_label.pkl")
    joblib.dump(model, model_path)
    print(f"Saved model to: {model_path}")

print("\n✅ All models trained and saved successfully.")
