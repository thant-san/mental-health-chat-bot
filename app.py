import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Cache the model and tokenizer to improve performance
@st.cache_resource
def load_model():
    model_name = "thantsan/mental_health_finetuned"  # Replace with your Hugging Face model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Load the model
st.title("Mental Health Chatbot")
tokenizer, model = load_model()

# Create input for user text
user_input = st.text_area("Enter your message:")

if st.button("Get Response"):
    if user_input:
        # Tokenize and predict
        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model(**inputs)
        prediction = outputs.logits.argmax(dim=-1).item()
        st.write(f"Prediction: {prediction}")
    else:
        st.warning("Please enter a message!")
