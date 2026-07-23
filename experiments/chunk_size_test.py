import sys
sys.path.insert(0, "src")
from chunking import load_all_sources
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from ingest import _get_embeddings
from langchain_core.vectorstores import InMemoryVectorStore

TEST_QUERIES = [
    ("What percentage of the US population is confident in their governments ability to regulate AI",
     "What is the number of documented AI insicdents from last year?",  
     "What percentage of experts espect a positive impact form AI?"
     "What model out-performed the top U.S. model in February 2025, and which country was that model from?"
     "SWE-bench")
]

for chunk_size in [300, 600, 1000]:
    docs = load_all_sources("data")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap= chunk_size // 8)
    chunks = splitter.split_documents(docs)

    store = InMemoryVectorStore(embedding=_get_embeddings())
    store.add_documents(chunks)
    print(f"\n chunk_size={chunk_size} ({len(chunks)} chunks)")
    for query, expected in TEST_QUERIES:
        results = store.similarity_search(query, k=3)
        found = any(expected.lower() in r.page_content.lower() for r in results)
        print(f" {query!r} -> expected term found: {found}")