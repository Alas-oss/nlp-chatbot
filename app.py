import streamlit as st
import sys
sys.path.insert(0, "src")
from dialogue_manager import get_response

st.title("NLP Chatbot")
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Say something -> "):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = get_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistand").write(response)