import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict/"  # Update with deployed FastAPI URL

st.title("Mental Health Chatbot")

user_input = st.text_input("You:", placeholder="Type your message here...")

if st.button("Send"):
    if user_input:
        try:
            response = requests.post(API_URL, json={"text": user_input})
            if response.status_code == 200:
                bot_response = response.json().get("response", "I didn't understand that.")
                st.write(f"🤖 Bot: {bot_response}")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to the API: {e}")
    else:
        st.warning("Please type a message!")
