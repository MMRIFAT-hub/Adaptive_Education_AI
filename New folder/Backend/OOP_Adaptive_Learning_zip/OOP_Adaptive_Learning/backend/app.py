import sys
import os
sys.path.append(os.path.abspath(".."))

from utils.co_topic_mapping import CO_TOPIC_MAP
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import openai
import os
import json
import networkx as nx  # Add this new import

# ✅ Build Knowledge Graph
G = nx.DiGraph()
G.add_edge("Principles of OOP", "Classes & Objects")
G.add_edge("Classes & Objects", "Inheritance & Polymorphism")
G.add_edge("Inheritance & Polymorphism", "Collections & Generics")

# ✅ Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ Initialize Flask
app = Flask(__name__)
CORS(app) 

# ✅ Load Student Profiles
with open("../data/student_profiles.json", "r") as f:
    student_profiles = json.load(f)

# ✅ Video Recommendations
recommendations = {
    "Principles of OOP": [
        "https://www.youtube.com/watch?v=TBWX97e1E9g",
        "https://www.youtube.com/watch?v=pTB0EiLXUC8"
    ],
    "Classes & Objects": [
        "https://www.youtube.com/watch?v=pi6eSNTgXbI"
    ],
    "Inheritance & Polymorphism": [
        "https://www.youtube.com/watch?v=YcX6AN1aF6Q"
    ],
    "OOP Core Concepts": [
        "https://www.youtube.com/watch?v=siA8I0zFz0g"
    ],
    "Collections & Generics": [
        "https://www.youtube.com/watch?v=Y2UhK1yzT3Q"
    ]
}

# ✅ Helper Functions
def get_recommendations(topic):
    personalized_videos = []

    # ✅ Add videos for the main topic
    if topic in recommendations:
        personalized_videos.extend(recommendations[topic])

    # ✅ Add videos for prerequisites from KG
    prereqs = list(G.predecessors(topic))
    for p in prereqs:
        if p in recommendations:
            personalized_videos.extend(recommendations[p])

    # ✅ Limit to 3 recommendations
    return personalized_videos[:3]


def generate_question(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an expert in Java programming and teaching."},
                      {"role": "user", "content": prompt}]
        )
        
        # Assuming response structure from GPT is something like this
        question_data = response.choices[0].message.content.strip()
        
        # Example of returning structured question with options and correct answer
        questions = []
        # Parse the question data and convert it to the desired structure
        # Assuming the API returns questions in this format:
        question = {
            "question": "What is OOP?",
            "options": ["Encapsulation", "Inheritance", "Polymorphism", "Abstraction"],
            "answer": "Encapsulation"
        }
        
        questions.append(question)
        
        return questions  # Return list of question dictionaries
        
    except Exception as e:
        return f"Error: {e}"



# ✅ Prompts
def generate_prompt(topic, difficulty, question_type="MCQ", prereqs=None):
    prereq_text = ""
    if prereqs:
        prereq_text = f"Also, ensure the question touches on these prerequisite topics: {', '.join(prereqs)}."

    if question_type == "MCQ":
        return f"""
        Generate a {difficulty} level multiple-choice question on Java for the topic: {topic}.
        {prereq_text}
        Requirements:
        - 1 correct answer and 3 wrong options
        - Indicate the correct answer
        Example:
        Q: <Question>
        A) ...
        B) ...
        C) ...
        D) ...
        Correct Answer: B
        """
    else:
        return f"""
        Generate a {difficulty} level Java coding problem on {topic}.
        {prereq_text}
        Include:
        - Problem statement
        - Expected output example
        """

def generate_teaching_prompt(topic, level):
    return f"""
    Teach the topic '{topic}' to a {level} student.
    Include:
    - A short definition
    - Why it is important in Object-Oriented Programming
    - At least one example in Java
    - If possible, an ASCII diagram
    - A short summary with 3 key points
    Format it in Markdown with headings for clarity.
    """

# ✅ Helper: Get next topic using KG and student profile
def get_next_topic_kg(profile, G):
    # Prioritize weak topics
    for topic in profile["weak_topics"]:
        if topic in G.nodes():
            prereqs = list(G.predecessors(topic))
            for p in prereqs:
                if p not in profile["strong_topics"]:
                    return p  # Teach prerequisite first
            return topic

    # If no weak topics left, check moderate
    for topic in profile["moderate_topics"]:
        if topic in G.nodes():
            return topic

    return None  # All mastered


# ✅ API Endpoints
@app.route("/get_profile/<int:student_id>", methods=["GET"])
def get_profile(student_id):
    profiles_file = "../data/student_profiles.json"

    try:
        with open(profiles_file, "r") as f:
            profiles = json.load(f)
    except FileNotFoundError:
        return jsonify({"error": "Profiles file not found"}), 404

    for profile in profiles:
        if profile["student_id"] == student_id:
            return jsonify(profile), 200

    return jsonify({"error": "Student not found"}), 404


@app.route("/get_next_topic", methods=["POST"])
def get_next_topic():
    data = request.json
    student_id = data["student_id"]
    level = data["level"]
    profile = next((s for s in student_profiles if s["student_id"] == student_id), None)

    if not profile:
        return jsonify({"error": "Student not found"}), 404

    topics = profile.get(
        "weak_topics" if level == "Level 1" else
        "moderate_topics" if level == "Level 2" else
        "strong_topics"
    )

    if not topics:
        return jsonify({"message": "No topics available at this level"})

    topic = topics[0]
    videos = get_recommendations(topic)

    return jsonify({"topic": topic, "videos": videos})


@app.route("/generate_questions", methods=["POST"])
def generate_questions():
    data = request.json
    topic = data["topic"]
    level = data["level"]
    
    # Determine difficulty based on level
    difficulty = "Easy" if level == "Level 1" else "Medium" if level == "Level 2" else "Hard"
    
    # Generate prompt for MCQ and coding question
    mcq_prompt = generate_prompt(topic, difficulty, "MCQ")
    mcq = generate_question(mcq_prompt)
    
    # Example of returning structured questions
    return jsonify({
        "topic": topic,
        "difficulty": difficulty,
        "MCQ": mcq  # Array of questions, each containing a question, options, and answer
    })


@app.route("/submit_score", methods=["POST"])
def submit_score():
    data = request.json
    score = data["score"]
    mastery = score >= 70
    return jsonify({"mastery": mastery, "message": "Passed" if mastery else "Retry required"})

# ✅ New Teaching Endpoint
@app.route("/teach_topic", methods=["POST"])
def teach_topic():
    data = request.json
    topic = data.get("topic")
    level = data.get("level", "Beginner")

    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    teaching_prompt = generate_teaching_prompt(topic, level)
    lesson = generate_question(teaching_prompt)

    return jsonify({
        "topic": topic,
        "lesson": lesson
    })

@app.route("/upload_co_marks", methods=["POST"])
def upload_co_marks():
    data = request.json
    student_id = data.get("student_id")

    if not student_id:
        return jsonify({"error": "Student ID required"}), 400

    # Thresholds for classification
    weak_threshold = 10
    moderate_threshold = 15
    weak_topics, moderate_topics, strong_topics = [], [], []

    # Convert CO marks to topic-based profile
    for co, topics in CO_TOPIC_MAP.items():
        score = data.get(co)
        if score is None:
            continue
        if score < weak_threshold:
            weak_topics.extend(topics)
        elif score < moderate_threshold:
            moderate_topics.extend(topics)
        else:
            strong_topics.extend(topics)

    # Create profile
    profile = {
        "student_id": int(student_id),
        "weak_topics": weak_topics,
        "moderate_topics": moderate_topics,
        "strong_topics": strong_topics
    }

    # Save to student_profiles.json
    profiles_file = "../data/student_profiles.json"
    try:
        with open(profiles_file, "r") as f:
            profiles = json.load(f)
    except FileNotFoundError:
        profiles = []

    # Update existing student profile or add new
    profiles = [p for p in profiles if p["student_id"] != int(student_id)]
    profiles.append(profile)

    with open(profiles_file, "w") as f:
        json.dump(profiles, f, indent=4)

    return jsonify({"message": "Profile updated successfully", "profile": profile})

@app.route("/teach_next_topic/<int:student_id>", methods=["GET"])
def teach_next_topic(student_id):
    profiles_file = "../data/student_profiles.json"
    try:
        with open(profiles_file, "r") as f:
            profiles = json.load(f)
    except FileNotFoundError:
        return jsonify({"error": "Profiles file not found"}), 404

    profile = next((p for p in profiles if p["student_id"] == student_id), None)
    if not profile:
        return jsonify({"error": "Student not found"}), 404

    # ✅ Determine next topic using KG
    next_topic = get_next_topic_kg(profile, G)
    if not next_topic:
        return jsonify({"message": "Student has mastered all topics"}), 200

    # ✅ Personalization logic
    if next_topic in profile["weak_topics"]:
        level = "Beginner"
        style = "Explain step-by-step with simple examples."
    elif next_topic in profile["moderate_topics"]:
        level = "Intermediate"
        style = "Use real-world examples and diagrams."
    else:
        level = "Advanced"
        style = "Provide quick summary with advanced coding challenges."

    # ✅ GPT Prompt for teaching
    prompt = f"""
    Teach '{next_topic}' for a {level} student.
    {style}
    Include:
    - Key points
    - At least one Java code example
    - Use Markdown for formatting
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful Java OOP instructor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        teaching_content = response.choices[0].message.content.strip()

    except Exception as e:
        teaching_content = f"Error generating GPT response: {e}"

    return jsonify({
        "student_id": student_id,
        "next_topic": next_topic,
        "level": level,
        "teaching": teaching_content
    })


# ✅ Run App
if __name__ == "__main__":
    app.run(debug=True)