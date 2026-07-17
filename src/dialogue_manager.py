import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from intent_classifier import predict
from entity_extractor import extract_entities

load_dotenv()

MODEL_PATH = "intent_model.pkl"
llm_model = init_chat_model("groq:openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"))


def get_response(user_input: str) -> str:
    response, confidence = predict(user_input, MODEL_PATH)
    entities = extract_entities(user_input)

    if response:
        print(f"[matched intent, confidence={confidence:.2f}, entities={entities}]")
        return response

    print(f"[low confidence ({confidence:.2f}), falling back to LLM]")
    result = llm_model.invoke(user_input)
    return result.content