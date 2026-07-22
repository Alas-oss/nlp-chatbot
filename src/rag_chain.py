import os
import sys
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from retriever import build_hybrid_retriever

load_dotenv()

langfuse = get_client()
langfuse_handler = CallbackHandler()

PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below. If the context doesn't
contain the answer, say you don't have that information rather than guessing.

Context:
{context}

Question: {question}

Answer:"""
)

def build_rag_chain():
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("Error: QROG_API_KEY not set in .env.")
        sys.exit(1)

    retriever = build_hybrid_retriever()
    model = init_chat_model("groq:openai/gpt-oss-120b", api_key=groq_key)
    chain = PROMPT | model

    def answer(question: str) -> dict:
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        try:
            result = chain.invoke(
                {"context": context, "question": question},
                config={
                    "callbacks": [langfuse_handler],
                    "metadata": {
                        "num_retrieved_chunks": len(docs),
                        "question_length": len(question),
                    },
                },
            )
            return {"answer": result.content, "sources": [d.page_content[:100] for d in docs]}
        except Exception as e:
            return {"answer": f"I ran into an error while generating a response: {e}", "sources": []}
        
    return answer