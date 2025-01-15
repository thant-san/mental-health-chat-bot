import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# Title of the app
st.title("Mental Health Chatbot")
st.write("A chatbot powered by your fine-tuned model for mental health guidance.")

# Load model and tokenizer
@st.cache_resource
def load_model():
    # Access Hugging Face token from environment variables
    hf_token = st.secrets["HUGGINGFACE_TOKEN"]  # Ensure this is added in secrets.toml

    # Hugging Face model details
    base_model_name = "unsloth/mistral-small-instruct-2409-bnb-4bit"
    fine_tuned_model_name = "thantsan/mental_health_finetuned"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=hf_token)

    # Load base model and apply PEFT (LoRA) fine-tuning
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, token=hf_token)
    model = PeftModel.from_pretrained(base_model, fine_tuned_model_name, token=hf_token)

    return tokenizer, model

# Initialize model and tokenizer
try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# User input
user_input = st.text_input("You:", placeholder="Type your message here...")

if st.button("Send"):
    if user_input:
        try:
            # Tokenize input
            inputs = tokenizer(user_input, return_tensors="pt")

            # Generate response
            outputs = model.generate(
                inputs["input_ids"],
                max_length=150,  # Set a limit for response length
                num_beams=5,     # Optional: Use beam search for better responses
                no_repeat_ngram_size=2,  # Avoid repetitive responses
                early_stopping=True,
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.write(f"🤖 Bot: {response}")
        except Exception as e:
            st.error(f"Error during response generation: {e}")
    else:
        st.warning("Please type a message!")

# Footer
st.caption("Powered by a fine-tuned LoRA model.")
