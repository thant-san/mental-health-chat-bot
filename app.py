import steamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import Torch

@st.cache_resource
def load_model():
    model_name = "thantsan/mental_health_finetuned"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

st.title("Mental Health Chatbot")
user_input = st.text_input("You:", placeholder="Type your message...")

if user_input:
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"])
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write(f"🤖 Bot: {response}")
