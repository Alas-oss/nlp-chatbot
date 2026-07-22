import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "groq:openai/gpt-oss-120b"

EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "groq:openai/gpt-oss-120b"

VECTOR_STORE_PATH = "vector_store.json"
CHUNKS_PATH = "chunks.json"
DATA_DIR = "data"

RETRIEVAL_K = 4
SEMANTIC_WEIGHT = 0.6
BM25_WEIGHT = 0.4

CHUNK_SIZE = 600
CHUNK_OVERLAP = 80


# INTENT_MODEL_PATH = "intent_model.pkl"
# CONFIDENCE_THRESHOLD = float(os.getnev("CONFIDENCE_THRESHOLD", "0.4"))