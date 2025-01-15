import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Cache the model and tokenizer
@st.cache_resource
def load_model():
    model_name = "thantsan/mental_health_finetuned"  # Replace with your model's path or name on Hugging Face
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    
    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN environment variable is not set.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
    return tokenizer, model

# Load the model and tokenizer
tokenizer, model = load_model()

# Streamlit app title
st.title("Custom Chatbot with Fine-Tuned Model")

# User input section
user_input = st.text_input("You:", placeholder="Type your message here...")

# Handle user input and generate a response
if st.button("Send"):
    if user_input.strip():  # Ensure input is not empty or whitespace
        # Tokenize user input
        inputs = tokenizer(user_input, return_tensors="pt")
        
        # Generate a response
        outputs = model.generate(
            inputs["input_ids"],
            max_length=100,  # Adjust max_length based on your model
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id  # Handle padding token
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Display the bot's response
        st.write(f"🤖 Bot: {response}")
    else:
        st.warning("Please type a message!")
