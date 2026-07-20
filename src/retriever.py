from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from ingest import load_vector_store
from chunking import load_docx_as_documents, chunk_documents

SOURCE_DOC_PATH = "data/kings_college_london.docx"


def build_hybrid_retriever():
    vector_store = load_vector_store()
    semantic_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    docs = load_docx_as_documents(SOURCE_DOC_PATH)
    chunks = chunk_documents(docs)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4

    return EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )