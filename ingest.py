import sys
sys.path.insert(0, "src")
from ingest import build_vector_store, save_chunks
from chunking import load_all_sources, chunk_documents

docs = load_all_sources("data")
chunks = chunk_documents(docs)
build_vector_store(chunks)
save_chunks(chunks)
print(f"Ingest {len(chunks)} chunks from all sources in data/")