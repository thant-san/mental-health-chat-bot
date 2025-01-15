import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource
def load_model():
    model_name = "bert-base-uncased"
    hf_token = os.getenv("HUGGINGFACE_TOKE")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModel.from_pretrained(model_name, use_auth_token=hf_token)
    return tokenizer, model

tokenizer, model = load_model()

st.title("Hugging Face Chatbot")

# User input
user_input = st.text_input("You:", placeholder="Type your message here...")

if st.button("Send"):
    if user_input:
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model.generate(inputs["input_ids"])
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write(f"🤖 Bot: {response}")
    else:
        st.warning("Please type a message!")
