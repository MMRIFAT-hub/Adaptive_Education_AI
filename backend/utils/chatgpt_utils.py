# File: app/services/chatgpt_utils.py

from youtubesearchpython import VideosSearch
import traceback
import os
from openai import OpenAI
from dotenv import load_dotenv
import re

print("chatgpt_utils.py loaded — os imported:", 'os' in globals())


# Load API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_lesson(topic: str, level: str) -> str:
    prompt = (
        f"You are an AI tutor. Create a comprehensive lesson on '{topic}' for a student at level {level}.\n\n"
        f"The lesson should prepare the student for three difficulty levels:\n"
        f"- Level 1: Basic concepts and definitions\n"
        f"- Level 2: Application-based problems\n" 
        f"- Level 3: Complex problem-solving and critical thinking\n\n"
        f"Include:\n"
        f"1. Clear explanations of fundamental concepts\n"
        f"2. Practical examples and code snippets\n"
        f"3. Real-world applications\n"
        f"4. Common pitfalls to avoid\n"
        f"5. Tips for approaching problems at different difficulty levels\n\n"
        f"Structure the lesson to progressively build from basic to advanced concepts."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful educational assistant that creates structured, comprehensive lessons."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating lesson for {topic}: {str(e)}"


def generate_quiz(topic: str, level: str, quiz_level: str = "level1") -> dict:
    """
    Generate a quiz for a specific topic, proficiency level, and difficulty level
    
    Args:
        topic (str): The programming topic
        level (str): Student's proficiency level
        quiz_level (str): Quiz difficulty level - "level1", "level2", or "level3"
    
    Returns:
        dict: Structured quiz data with questions, options, and answers
    """
    
    # Define difficulty parameters based on quiz level
    level_config = {
        "level1": {
            "difficulty": "basic",
            "questions": 3,
            "description": "basic concepts and definitions",
            "hint": "Focus on fundamental concepts and simple applications. Keep questions straightforward."
        },
        "level2": {
            "difficulty": "intermediate", 
            "questions": 4,
            "description": "application-based problems",
            "hint": "Include scenario-based questions and moderate complexity. Test understanding of concepts."
        },
        "level3": {
            "difficulty": "advanced",
            "questions": 5,
            "description": "complex problem-solving and critical thinking",
            "hint": "Challenge with complex scenarios, code analysis, and edge cases. Include reasoning questions."
        }
    }
    
    config = level_config[quiz_level]
    
    prompt = f"""
    Generate a {config['difficulty']} level quiz about {topic} for a student with proficiency level {level}.
    
    Requirements:
    - Create exactly {config['questions']} multiple-choice questions
    - Difficulty: {config['difficulty']} - {config['description']}
    - {config['hint']}
    - Each question must have 4 options labeled A, B, C, D
    - Format the response as valid JSON with this exact structure:
    
    {{
        "quiz_title": "string",
        "topic": "string",
        "difficulty_level": "{quiz_level}",
        "proficiency_level": "{level}",
        "questions": [
            {{
                "question_number": 1,
                "question_text": "string",
                "options": {{
                    "A": "option text",
                    "B": "option text", 
                    "C": "option text",
                    "D": "option text"
                }},
                "correct_answer": "A",
                "explanation": "string explaining why this is correct"
            }}
        ]
    }}
    
    Important: Return ONLY the JSON object, no additional text or explanations.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful quiz generator that always returns valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        quiz_content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        import json
        quiz_data = json.loads(quiz_content)
        
        # Add metadata
        quiz_data["metadata"] = {
            "total_questions": len(quiz_data["questions"]),
            "expected_questions": config["questions"]
        }
        
        return quiz_data
        
    except Exception as e:
        # Return a structured error response
        return {
            "quiz_title": f"{topic} Quiz - {quiz_level}",
            "topic": topic,
            "difficulty_level": quiz_level,
            "proficiency_level": level,
            "questions": [],
            "error": f"Failed to generate quiz: {str(e)}",
            "metadata": {
                "total_questions": 0,
                "expected_questions": config["questions"]
            }
        }
