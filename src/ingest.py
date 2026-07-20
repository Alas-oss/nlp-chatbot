from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv()

VECTOR_STORE_PATH = "vector_store.json"

def _get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

def build_vector_store(chunks):
    store = InMemoryVectorStore(embedding=_get_embeddings())
    store.add_documents(chunks)
    store.dump(VECTOR_STORE_PATH)
    return store

def load_vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore.load(VECTOR_STORE_PATH, embedding=_get_embeddings())