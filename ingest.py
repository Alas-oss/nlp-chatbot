import sys
sys.path.insert(0, "src")
from ingest import build_vector_store, save_chunks
from chunking import load_docx_as_documents, chunk_documents

docs = load_docx_as_documents("data/kings_college_london.docx")
chunks = chunk_documents(docs)
build_vector_store(chunks)
save_chunks(chunks)
print(f"Ingest {len(chunks)} chunks: vector store + chunks.json written")
