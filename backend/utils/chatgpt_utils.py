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
        f"You are an AI tutor. Create a detailed, beginner-friendly lesson on the topic '{topic}'. "
        f"Assume the student is at the '{level}' level. Include examples and simple explanations."
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful educational assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def generate_quiz(topic: str, level: str) -> str:
    prompt = (
        f"Generate 3 multiple-choice quiz questions on the topic '{topic}' for a student at '{level}' level. "
        f"Each question should have 4 options labeled A–D and MUST follow this answer format: "
        f"-[Correct Answer: C] at the end of each question. Example:\n"
        f"Question 1: ...\nA) ...\nB) ...\nC) ...\nD) ...\n-[Correct Answer: C]"
    )   

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful quiz generator."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()
