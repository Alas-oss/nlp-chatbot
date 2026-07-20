import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "groq:openai/gpt-oss-120b"
INTENT_MODEL_PATH = "intent_model.pkl"
CONFIDENCE_THRESHOLD = float(os.getnev("CONFIDENCE_THRESHOLD", "0.4"))