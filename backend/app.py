from utils.chatgpt_utils import generate_lesson, generate_quiz
from utils.predictor import predict_student_levels
import json
from openai import OpenAI
import os
import firebase_admin
from firebase_admin import credentials, db
from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this import
import joblib
import logging
import numpy as np

# Initialize ChatGPT client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Firebase Admin SDK
cred = credentials.Certificate(r'../config/edora2-firebase-adminsdk-fbsvc-99ffb3c4a7.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://edora2-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# Initialize Flask app
app = Flask(__name__)

# Configure CORS - Add this section
CORS(app, resources={
    r"/*": {
        "origins": ["http://127.0.0.1:5500", "http://localhost:5500", "http://localhost:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Add CORS headers manually for OPTIONS requests
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://127.0.0.1:5500')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Handle OPTIONS requests for CORS preflight
@app.route('/predict_level', methods=['OPTIONS'])
@app.route('/generate_chatgpt_content', methods=['OPTIONS'])
@app.route('/chatbot', methods=['OPTIONS'])
def options_response():
    return '', 200

# Load pre-trained models with error handling
try:
    constructor_model = joblib.load('models/rf_model_Constructor_label.pkl')
    encapsulation_model = joblib.load('models/rf_model_Encapsulation_label.pkl')
    inheritance_model = joblib.load('models/rf_model_Inheritance_label.pkl')
    interface_model = joblib.load('models/rf_model_Interface_label.pkl')
    polymorphism_model = joblib.load('models/rf_model_Polymorphism_label.pkl')
    logging.info("All models loaded successfully")
    
    # Debug: Check model information
    logging.info(f"Constructor model type: {type(constructor_model)}")
    if hasattr(constructor_model, 'n_features_in_'):
        logging.info(f"Constructor model expects {constructor_model.n_features_in_} features")
    
except Exception as e:
    logging.error(f"Error loading models: {str(e)}")
    raise

logging.basicConfig(level=logging.DEBUG)

def get_student_data(student_identifier):
    """Get student data by email or student_id from students node"""
    try:
        ref = db.reference('students')
        all_students = ref.get()
        
        if not all_students:
            raise ValueError("No students found in database")
        
        logging.debug(f"Student keys in database: {list(all_students.keys())}")
        
        # Search through all students
        for student_key, student_data in all_students.items():
            # Check if this is the student we're looking for
            if (student_data.get('email') == student_identifier or 
                student_data.get('student_id') == student_identifier or
                student_data.get('userId') == student_identifier):
                
                logging.debug(f"Found student with key: {student_key}")
                return student_data
        
        # If we get here, student wasn't found
        available_emails = []
        available_student_ids = []
        
        for student_key, student_data in all_students.items():
            if isinstance(student_data, dict):
                if student_data.get('email'):
                    available_emails.append(student_data.get('email'))
                if student_data.get('student_id'):
                    available_student_ids.append(student_data.get('student_id'))
        
        raise ValueError(
            f"No student found with identifier: {student_identifier}\n"
            f"Available emails: {available_emails}\n"
            f"Available student_ids: {available_student_ids}"
        )
        
    except Exception as e:
        logging.error(f"Error fetching student data: {str(e)}")
        raise

def predict_proficiency(model, student_data, topic_name):
    """Predict proficiency level using student data with better error handling"""
    try:
        # Extract features from student data
        topic_scores = student_data.get('topic_scores', {})
        
        # Method 1: Try using just the specific topic score
        specific_score = topic_scores.get(f'{topic_name}_label', 0)
        features_method1 = [specific_score]
        
        # Method 2: Try using all topic scores
        features_method2 = [
            topic_scores.get('Class_label', 0),
            topic_scores.get('Constructor_label', 0),
            topic_scores.get('Encapsulation_label', 0),
            topic_scores.get('Inheritance_label', 0),
            topic_scores.get('Interface_label', 0),
            topic_scores.get('Polymorphism_label', 0)
        ]
        
        # Method 3: Try using topic scores + other metrics
        features_method3 = features_method2 + [
            student_data.get('midtermScore', 0),
            student_data.get('percentage', 0),
            student_data.get('score', 0)
        ]
        
        # Try different feature sets
        feature_sets = [
            features_method1,
            features_method2, 
            features_method3
        ]
        
        for i, features in enumerate(feature_sets, 1):
            try:
                logging.debug(f"Trying feature set {i} for {topic_name}: {features}")
                
                # Convert to numpy array and ensure correct shape
                features_array = np.array(features).reshape(1, -1)
                logging.debug(f"Features array shape: {features_array.shape}")
                
                # Check if model has n_features_in_ attribute
                if hasattr(model, 'n_features_in_'):
                    expected_features = model.n_features_in_
                    actual_features = features_array.shape[1]
                    logging.debug(f"Model expects {expected_features} features, we have {actual_features}")
                    
                    if expected_features != actual_features:
                        logging.warning(f"Feature mismatch for {topic_name}: expected {expected_features}, got {actual_features}")
                        continue  # Try next feature set
                
                prediction = model.predict(features_array)
                logging.info(f"Successfully predicted {topic_name} with feature set {i}: {prediction[0]}")
                return int(prediction[0])
                
            except Exception as e:
                logging.warning(f"Feature set {i} failed for {topic_name}: {str(e)}")
                continue
        
        # If all feature sets fail
        logging.error(f"All feature sets failed for {topic_name}")
        return None
        
    except Exception as e:
        logging.error(f"Error in prediction for {topic_name}: {str(e)}")
        return None

@app.route('/check_connection', methods=['GET'])
def check_connection():
    try:
        ref = db.reference('/')
        data = ref.get()
        if data is not None:
            student_count = len(data.get('students', {})) if data.get('students') else 0
            teacher_count = len(data.get('teachers', {})) if data.get('teachers') else 0
            
            return jsonify({
                "status": "success", 
                "message": "Connected to Firebase!",
                "student_count": student_count,
                "teacher_count": teacher_count,
                "database_structure": list(data.keys()) if data else []
            }), 200
        else:
            return jsonify({"status": "error", "message": "Failed to retrieve data from Firebase!"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to connect to Firebase: {str(e)}"}), 500

@app.route('/debug_models', methods=['GET'])
def debug_models():
    """Endpoint to debug model information"""
    try:
        models_info = {}
        topic_models = {
            'Constructor': constructor_model,
            'Encapsulation': encapsulation_model,
            'Inheritance': inheritance_model,
            'Interface': interface_model,
            'Polymorphism': polymorphism_model
        }
        
        for topic, model in topic_models.items():
            model_info = {
                'type': str(type(model)),
                'module': str(model.__class__.__module__)
            }
            
            if hasattr(model, 'n_features_in_'):
                model_info['expected_features'] = model.n_features_in_
            if hasattr(model, 'feature_names_in_'):
                model_info['feature_names'] = list(model.feature_names_in_)
            if hasattr(model, 'n_classes_'):
                model_info['n_classes'] = model.n_classes_
                
            models_info[topic] = model_info
        
        return jsonify({
            'status': 'success',
            'models_info': models_info
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_level', methods=['POST'])
def predict_level():
    try:
        data = request.get_json()
        if not data or 'student_id' not in data:
            return jsonify({'error': 'Missing student_id in request'}), 400
            
        student_identifier = data['student_id']
        logging.debug(f"Received prediction request for: {student_identifier}")

        # Fetch student data
        student_data = get_student_data(student_identifier)
        logging.debug(f"Student data: {student_data}")
        
        # Define topics and their corresponding models
        topic_models = {
            'Constructor': constructor_model,
            'Encapsulation': encapsulation_model,
            'Inheritance': inheritance_model,
            'Interface': interface_model,
            'Polymorphism': polymorphism_model
        }
        
        predictions = {}
        prediction_details = {}
        
        for topic, model in topic_models.items():
            try:
                proficiency_level = predict_proficiency(model, student_data, topic)
                predictions[topic] = proficiency_level
                prediction_details[topic] = {
                    'success': proficiency_level is not None,
                    'current_score': student_data.get('topic_scores', {}).get(f'{topic}_label', 'N/A')
                }
                
            except Exception as e:
                logging.error(f"Error predicting {topic}: {str(e)}")
                predictions[topic] = None
                prediction_details[topic] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Prepare response
        response = {
            'student_id': student_identifier,
            'student_name': student_data.get('name', 'Unknown'),
            'email': student_data.get('email', 'Unknown'),
            'predictions': predictions,
            'prediction_details': prediction_details,
            'current_scores': student_data.get('topic_scores', {}),
            'overall_percentage': student_data.get('percentage', 0),
            'midterm_score': student_data.get('midtermScore', 0)
        }
        
        logging.debug(f"Final response: {response}")
        return jsonify(response), 200

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/student/<student_identifier>', methods=['GET'])
def get_student_info(student_identifier):
    """Endpoint to get student information by email, student_id, or userId"""
    try:
        student_data = get_student_data(student_identifier)
        return jsonify({
            'status': 'success',
            'student_data': student_data
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/all_students', methods=['GET'])
def get_all_students():
    """Endpoint to get all students (for debugging)"""
    try:
        ref = db.reference('students')
        students_data = ref.get()
        
        students = []
        if students_data:
            for key, data in students_data.items():
                if isinstance(data, dict):
                    students.append({
                        'firebase_key': key,
                        'email': data.get('email'),
                        'name': data.get('name'),
                        'student_id': data.get('student_id'),
                        'userId': data.get('userId')
                    })
        
        return jsonify({
            'status': 'success',
            'total_students': len(students),
            'students': students
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/generate_chatgpt_content', methods=['POST'])
def generate_chatgpt_content():
    try:
        data = request.get_json()
        student_id = data.get("student_id")
        predictions = data.get("predicted_levels")

        if not student_id or not predictions:
            return jsonify({'error': 'Missing student_id or predicted_levels'}), 400

        # 🔍 Find student in Firebase to get current scores
        students_ref = db.reference("students")
        all_students = students_ref.get()

        matching_key = None
        student_data = None
        for key, value in all_students.items():
            if value.get("email") == student_id or value.get("student_id") == student_id:
                matching_key = key
                student_data = value
                break

        if not matching_key:
            return jsonify({'error': f"Student with ID '{student_id}' not found in Firebase"}), 404

        # Get student's current topic scores
        topic_scores = student_data.get('topic_scores', {})
        result = {}

        for topic, level in predictions.items():
            # Check if student already mastered this topic (score >= 90)
            topic_score_key = f'{topic}_label'
            current_score = topic_scores.get(topic_score_key, 0)
            
            if current_score >= 90:
                print(f"Skipping {topic} - student already mastered with score: {current_score}")
                result[topic] = {
                    "level": level,
                    "lesson": "Already mastered - no lesson needed",
                    "quizzes": {
                        "level1": "Already mastered - no quiz needed",
                        "level2": "Already mastered - no quiz needed", 
                        "level3": "Already mastered - no quiz needed"
                    },
                    "already_mastered": True,
                    "current_score": current_score
                }
            else:
                print(f"Generating content for {topic} (Level {level}), current score: {current_score}")
                lesson = generate_lesson(topic, level)
                
                # Generate quizzes for all 3 levels
                quiz_level1 = generate_quiz(topic, level, "level1")
                quiz_level2 = generate_quiz(topic, level, "level2")
                quiz_level3 = generate_quiz(topic, level, "level3")

                result[topic] = {
                    "level": level,
                    "lesson": lesson,
                    "quizzes": {
                        "level1": quiz_level1,
                        "level2": quiz_level2,
                        "level3": quiz_level3
                    },
                    "already_mastered": False,
                    "current_score": current_score,
                    "quiz_progress": {
                        "level1": {"completed": False, "score": 0},
                        "level2": {"completed": False, "score": 0},
                        "level3": {"completed": False, "score": 0}
                    }
                }

        # ✅ Save to Firebase
        db.reference(f"students/{matching_key}/generated_content").set(result)

        return jsonify({
            "status": "success",
            "firebase_key": matching_key,
            "student_id": student_id,
            "saved_to_firebase": True,
            "generated_content": result
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to generate content: {str(e)}'}), 500


@app.route('/firebase_predict_levels', methods=['POST'])
def firebase_predict_levels():
    try:
        data = request.get_json()
        if not data or 'student_id' not in data:
            return jsonify({'error': 'Missing student_id in request'}), 400

        student_id = data['student_id']
        student_data = get_student_data(student_id)

        topic_scores = student_data.get('topic_scores', {})
        if not topic_scores:
            return jsonify({'error': 'No topic_scores found for this student'}), 400

        # Predict levels using your model
        predictions = predict_student_levels(student_id, topic_scores)

        return jsonify({
            'status': 'success',
            'student_id': student_id,
            'predicted_levels': predictions
        }), 200

    except Exception as e:
        return jsonify({'error': f'Failed to predict levels: {str(e)}'}), 500


@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.get_json()
        student_id = data.get("student_id")
        question = data.get("message")

        if not student_id or not question:
            return jsonify({'error': 'Missing student_id or message'}), 400

        # 🔍 Find student in Firebase
        students_ref = db.reference("students")
        all_students = students_ref.get()

        matching_student = None
        for key, value in all_students.items():
            if value.get("email") == student_id or value.get("student_id") == student_id:
                matching_student = value
                break

        if not matching_student:
            return jsonify({'error': f"No student found with ID: {student_id}"}), 404

        # 🧠 Inject student data into prompt
        context = f"""
You are an AI assistant helping a student. Here's their data:
- Name: {matching_student.get('name')}
- Email: {matching_student.get('email')}
- Score: {matching_student.get('score')}
- Predicted Levels: {matching_student.get('predicted_levels', {})}
- Topic Scores: {matching_student.get('topic_scores', {})}
- Generated Content: {list(matching_student.get('generated_content', {}).keys())}

Student Question: "{question}"
Respond helpfully and clearly.
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful academic assistant."},
                {"role": "user", "content": context}
            ]
        )

        answer = response.choices[0].message.content.strip()

        return jsonify({"response": answer}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to respond: {str(e)}'}), 500


@app.route('/update_quiz_progress', methods=['POST'])
def update_quiz_progress():
    try:
        data = request.get_json()
        student_id = data.get("student_id")
        topic = data.get("topic")
        quiz_level = data.get("quiz_level")
        score = data.get("score")
        completed = data.get("completed", False)

        if not all([student_id, topic, quiz_level]):
            return jsonify({'error': 'Missing required fields'}), 400

        # Find student
        students_ref = db.reference("students")
        all_students = students_ref.get()

        matching_key = None
        for key, value in all_students.items():
            if value.get("email") == student_id or value.get("student_id") == student_id:
                matching_key = key
                break

        if not matching_key:
            return jsonify({'error': f"Student with ID '{student_id}' not found"}), 404

        # Update quiz progress
        quiz_progress_path = f"students/{matching_key}/generated_content/{topic}/quiz_progress/{quiz_level}"
        db.reference(quiz_progress_path).update({
            "completed": completed,
            "score": score,
            "last_attempt": {".sv": "timestamp"}
        })

        # If student completes level 3 with good score, update topic score
        if quiz_level == "level3" and completed and score >= 80:
            new_topic_score = min(100, score)  # Cap at 100
            topic_score_path = f"students/{matching_key}/topic_scores/{topic}_label"
            db.reference(topic_score_path).set(new_topic_score)

        return jsonify({
            "status": "success",
            "message": f"Quiz progress updated for {topic} {quiz_level}",
            "score": score,
            "completed": completed
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to update progress: {str(e)}'}), 500


# Update CORS for Firebase Hosting
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:3000", 
            "http://localhost:5500",
            "https://edora2.web.app",  # Your Firebase URL
            "https://edora2.firebaseapp.com"  # Alternative Firebase URL
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Remove the app.run at the bottom and replace with:
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)