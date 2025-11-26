# utils/predictor.py

import os
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # .../backend
MODELS_DIR = os.path.join(BASE_DIR, "models")

TOPICS = ["Class", "Constructor", "Encapsulation", "Inheritance", "Interface", "Polymorphism"]

topic_models = {}

for topic in TOPICS:
    model_path = os.path.join(MODELS_DIR, f"rf_model_{topic}_label.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found for topic {topic}: {model_path}")
    topic_models[topic] = joblib.load(model_path)


def predict_student_levels(student_id, topic_scores):
    """
    Predict beginner(0) / intermediate(1) / advanced(2) for each topic.

    topic_scores is expected to be a dict like:
    {
        "Class_label":  0–100,
        "Constructor_label": 0–100,
        ...
    }
    """

    # Build a single 6-feature vector from the student's topic scores
    features = [
        topic_scores.get("Class_label", 0),
        topic_scores.get("Constructor_label", 0),
        topic_scores.get("Encapsulation_label", 0),
        topic_scores.get("Inheritance_label", 0),
        topic_scores.get("Interface_label", 0),
        topic_scores.get("Polymorphism_label", 0),
    ]

    X = np.array(features, dtype=float).reshape(1, -1)

    predictions = {}

    for topic, model in topic_models.items():
        y = model.predict(X)[0]  # 0 = beginner, 1 = intermediate, 2 = advanced
        predictions[topic] = int(y)

    return predictions
