import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer from Hugging Face
@st.cache_resource
def load_model():
    model_name = "your-huggingface-model-name"  # Replace with your model's name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model('thantsan/mental_health_finetuned')

st.title("Hugging Face Model Deployment")

# Input for user text
user_input = st.text_input("Enter text:")

if st.button("Predict"):
    if user_input:
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt")  # Change to "tf" if using TensorFlow
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1).item()  # Modify based on your model type
        st.write("Bot Response:", predictions)
    else:
        st.warning("Please enter text!")
