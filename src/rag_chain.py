import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from retriever import build_hybrid_retriever

load_dotenv()

PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below. If the context doesn't
contain the answer, say you don't have that information rather than guessing.

Context:
{context}

Question: {question}

Answer:"""
)


def build_rag_chain():
    retriever = build_hybrid_retriever()
    model = init_chat_model("groq:openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"))

    def answer(question: str) -> dict:
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        chain = PROMPT | model
        result = chain.invoke({"context": context, "question": question})
        return {"answer": result.content, "sources": [d.page_content[:100] for d in docs]}

    return answer