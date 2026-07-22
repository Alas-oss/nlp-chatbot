import sys
sys.path.insert(0, "src")
from dialogue_manager import get_response
from rag_chain import langfuse

print("NLP Chatbot ready. Type 'quit'/'exit' to exit.")
while True:
    user_input = input("> ").strip()
    if not user_input:
        continue
    if user_input.lower() in ("quit", "exit"):
        print("Goodbye!")
        langfuse.flush()
        break
    print(get_response(user_input))