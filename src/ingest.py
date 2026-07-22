import os
import sys
import json
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

load_dotenv()

VECTOR_STORE_PATH = "vector_store.json"
CHUNKS_PATH = "chunks.json"

def _get_embeddings():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        print("Error: GOOGLE_API_KEY not set in .env - required for embeddings.")
        sys.exit(1)
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

def build_vector_store(chunks):
    store = InMemoryVectorStore(embedding=_get_embeddings())
    store.add_documents(chunks)
    store.dump(VECTOR_STORE_PATH)
    return store

def load_vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore.load(VECTOR_STORE_PATH, embedding=_get_embeddings())

def save_chunks(chunks: list[Document]):
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump([{"content": c.page_content, "metadata": c.metadata} for c in chunks], f)

def load_chunks() -> list[Document]:
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return [Document(page_content=d["content"], metadata=d["metadata"]) for d in data]
