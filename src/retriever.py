from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from ingest import load_vector_store, load_chunks

def build_hybrid_retriever():
    vector_store = load_vector_store()
    semantic_retriever = vector_store.as_retriever(search_kwargs={"k":4})

    chunks = load_chunks()
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4

    return EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )