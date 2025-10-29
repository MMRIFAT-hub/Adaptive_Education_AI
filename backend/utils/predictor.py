# This is the main prediction logic
# File: app/services/predictor.py

import os
import json
from typing import Dict
import joblib


# Load saved models
MODEL_DIR = "models"
loaded_models = {}
loaded_encoders = {}

for file in os.listdir(MODEL_DIR):
    if file.startswith("rf_model_") and file.endswith(".pkl"):
        topic = file.replace("rf_model_", "").replace(".pkl", "")
        model_path = os.path.join(MODEL_DIR, file)
        encoder_path = os.path.join(MODEL_DIR, f"label_encoder_{topic}.pkl")

        # Load Random Forest model
        model = joblib.load(model_path)
        loaded_models[topic] = model

        # Load label encoder
        if os.path.exists(encoder_path):
            encoder = joblib.load(encoder_path)
            loaded_encoders[topic] = encoder


def predict_student_levels(student_id: str, topic_scores: Dict[str, float]) -> Dict[str, str]:
    predictions = {}
    
    # Build full feature vector once
    features = [
        topic_scores.get("Class_label", 0),
        topic_scores.get("Encapsulation_label", 0),
        topic_scores.get("Inheritance_label", 0),
        topic_scores.get("Polymorphism_label", 0),
        topic_scores.get("Constructor_label", 0),
        topic_scores.get("Interface_label", 0)
    ]

    for topic in topic_scores:
        if topic in loaded_models:
            model = loaded_models[topic]
            encoder = loaded_encoders.get(topic)

            level_numeric = model.predict([features])[0]
            if encoder:
                level = encoder.inverse_transform([int(level_numeric)])[0]
            else:
                level = str(level_numeric)

            predictions[topic] = level


    # Save to profile
    student_data = {
        "student_id": student_id,
        "topics": topic_scores,
        "predicted_levels": predictions
    }

    os.makedirs("data/student_profiles", exist_ok=True)
    with open(f"data/student_profiles/{student_id}.json", "w") as f:
        json.dump(student_data, f, indent=2)

    return predictions

