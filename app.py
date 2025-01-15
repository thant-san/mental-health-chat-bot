import streamlit as st

# Page configuration
st.set_page_config(page_title="Simple Chatbot", layout="centered")

# Title and subtitle
st.title("Mental Health Chatbot")
st.subheader("Have a conversation with our AI-powered assistant")

# User input box
user_input = st.text_input("You:", placeholder="Type your message here...", key="user_input")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add the user's message to the chat history
if user_input:
    st.session_state.messages.append({"sender": "You", "message": user_input})

    # Placeholder for chatbot response
    bot_response = f"I'm here to help! You said: {user_input}"
    st.session_state.messages.append({"sender": "Bot", "message": bot_response})

    # Clear the input box
    st.experimental_rerun()

# Display the chat history
for msg in st.session_state.messages:
    sender = "🧑" if msg["sender"] == "You" else "🤖"
    st.write(f"{sender} **{msg['sender']}**: {msg['message']}")

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()
