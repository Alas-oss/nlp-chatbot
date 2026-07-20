from rag_chain import build_rag_chain
from entity_extractor import extract_entities

_rag_answer = build_rag_chain() 


def get_response(user_input: str) -> str:
    entities = extract_entities(user_input)
    result = _rag_answer(user_input)
    print(f"[entities={entities}]")
    print(f"[retrieved {len(result['sources'])} source chunks]")
    return result["answer"]