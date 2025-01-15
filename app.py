import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Streamlit app setup
st.set_page_config(page_title="Mental Health Chatbot", layout="wide")
st.title("Mental Health Chatbot")
st.write("This chatbot is designed to assist with mental health-related queries.")

# Load model and tokenizer
@st.cache_resource
def load_model():
    # Access Hugging Face token from secrets
    hf_token = st.secrets["HUGGINGFACE_TOKEN"]

    # Base and fine-tuned model names
    base_model_name = "unsloth/mistral-small-instruct-2409-bnb-4bit"
    fine_tuned_model_name = "thantsan/mental_health_finetuned"

    # Initialize quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Use 4-bit quantization
        bnb_4bit_compute_dtype="float16",  # Set computation dtype
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=hf_token)

    # Load base model with quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        token=hf_token,
    )

    # Apply fine-tuning with PEFT
    model = PeftModel.from_pretrained(base_model, fine_tuned_model_name, token=hf_token)

    return tokenizer, model

# Load the model and tokenizer
tokenizer, model = load_model()

# Chat functionality
def generate_response(user_input):
    # Tokenize user input
    inputs = tokenizer(user_input, return_tensors="pt")
    # Generate response
    outputs = model.generate(inputs["input_ids"], max_length=200, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit app interface
st.text_input("Type your message here:", key="user_input", on_change=lambda: None)

if st.session_state.get("user_input"):
    user_message = st.session_state["user_input"]
    st.write(f"**You:** {user_message}")

    with st.spinner("Thinking..."):
        bot_response = generate_response(user_message)

    st.write(f"**Bot:** {bot_response}")
    st.session_state["user_input"] = ""  # Clear input after sending
