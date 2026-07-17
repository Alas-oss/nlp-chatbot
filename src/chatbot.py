import sys
sys.path.insert(0, "src")

from dialogue_manager import get_response

print("NLP Chatbot ready. Type 'quit' to exit.")
while True:
    user_input = input("> ").strip()
    if not user_input:
        continue
    if user_input.lower() in ("quit", "exit"):
        print("Goodbye!")
        break
    print(get_response(user_input))