# Project Report: RAG-Based Domain Chatbot

## 1. Purpose

This project originally used (TF_IDF + logistic regression, matching a small set of hand-written categories, falling back to an ungrounded LLM call for anything else). It ws deliberately rebuilt into a retrieval-augmented generation (RAG) system: instead of classifying a message into one of a few trained categories, the system retrieves relevant passages from a specific source document and has an LLM generate an answer grounded in that retrieved content. The goal shifted form "recognize a handful of known message types" to "answer open-ended questions accurately about a specific body of knowledge," which a fixed-category classifier structurally cannot do.

## 2. Why the pivot away from intent classification

The original classifier worked well for its narrow purpose (greetings, FAQ-style categories) but had no mechanism to "know" facts - it could only route a message to one of a small number of trained buckets. Expanding its scope to cover general knowledge would have meant either training on an impractically large, constantly-growing set of categories, or accepting that most questions would fall through to an ungrounded LLM call with no guarantee that the answer refelcted any particular source that is true. RAG addresses this directly: retrieval narrows the LLM's attention to the most relevant real content before it generates anything, and an explicit grouding instruction in the prompt tells it to decline rather than guess when the retrieved context doesn't contain an answer.

## 3. Architecture

- **Chunking**: a source `.docx` document is loaded and plit using a recursive character text splitter, which prefers paragraph and setence boundaries over hard character cuts, keeping each chunk topically coherent.
- **Embeddings + vector storage**: chunks are embedded via an API-based embedding model and stored in a vector index for similarity search. The vector store implementation went through two iterations
- **Hybrid retrieval**: queries are matched two ways simultaneously - semantic chunks - combined via reciprocal rank fusion (about 60% semantic / 40% keyword weighting). Semantic search generalizes over phrasing; BM25 reliably catches exact terms and proper nouns that embedding similarity can under-weight.
- **Grounded generation**: retrieved chunks are inserted into a prompt that explicitly instructs the model to answer only from that context and to say so plainly when it doesn't have the relevant information, rather than filling gaps form its own general training knowledge. 
- **Supplementary NER**: a lightweight names-entity-recognition pass runs alongside retrieval on the user's query, providing additional structured signal independent of the RAG pipeline itself.

## 4. Building process - technical challenges and how they were resolved

**Import-time crash from an unrelated dependency.** The text-splitter package's top-level import unconditionally pulled in a sentence-embedding library (and transitively, a deep learning framework), which crashed at import time on Windows before any of the actual chunking logic ran. Fixed by importing the needed class directly form its submodule rather than the package's `__init__`, bypassing the problematic import chain entirely - the chunking logic itself never needed that dependency.

**Local embedding model made unnecessary.** An initial design used a locally-run embedding model, which meant importing a deep learning runtime just to embed short string of text. This was replaced with an API-based embedding call instead, which changed to no local ML runtime, no large model download, and no exposure to platform-specific native-binary issues for that piece of the pipeline.

**A genuine native-binary crash in the vector store.** Building the vector index initially caused a hard access-violation crash (not a Python exception, but a segmentation fault in compiled C++ code) inside the chosen vector databse's native dependencies. This was isolated methodically: importing the database package alone worked, importing the embedding client alone worked, but building an actual index crashed. Since the underlying native library issue couldn't be resolved without administrator-level system changes that weren't available, the fix was to swap to a vector store implementation with zero compiled depedencies, using a pure-Python, Numpy-backed in-memory store. This trades some scalability at very large corpus sizes for complete reliability at the scale this project actually operates at, and eliminated as entire class of platform-specific crashed risk.

**Package recognition broke an import mid-project.** A retrieval utility class that combines multiple retrievers was removed from the core orchestration library's public namespace in a recent version and relocated to a separately version package. This was diagnosed by checking directly again the installed package version rather than assuming the older import path was still appropriate, and was comfirming that some functionality that used to live in the main library has been split out into companion packages as the library matures, and that verifying against the actual installed version, not memory or older documentation, id the reliable way to resolve these import errors.

## 5. Current status

- The full pipeline - chunking, embedding, hybrid retrieval, grounded generation - is built and verified working end to end: a built vector store can be persisted, reloaded, and queried, returning content-relevant chunks; the RAG chain correctly declines to answer questions outside its retrieved context rather than falling back to ungrounded general knowledge.
- The chatbot's interactive loop is unchanged in interface from the earlier classifier-based version - only the reponse-generation logic underneath it was replaced, so the entrypoint script required no modification.
- Open items: retrieval currently covers a single source document; extending to multiple documents would need per-chunk source metadata so answers can be traced back to their originating file. Chunk size/count and retrieval breadth are also tunable levers for expanding the range of questions the system can answer well, without requiring any artchitectural change.