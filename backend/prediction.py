import joblib

# Load the models from disk
constructor_model = joblib.load('models/rf_model_Constructor_label.pkl')
encapsulation_model = joblib.load('models/rf_model_Encapsulation_label.pkl')
inheritance_model = joblib.load('models/rf_model_Inheritance_label.pkl')
interface_model = joblib.load('models/rf_model_Interface_label.pkl')
polymorphism_model = joblib.load('models/rf_model_Polymorphism_label.pkl')


def predict_proficiency(model, student_data):
    """
    Predict the proficiency level of the student based on their data.
    """
    features = [student_data['score']]  # Assuming 'score' is a relevant feature
    prediction = model.predict([features])
    return prediction[0]  # Returns the predicted proficiency level
