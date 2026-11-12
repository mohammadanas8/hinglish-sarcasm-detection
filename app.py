
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "mohammadanas8/hinglish-sarcasm-mbert"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("Hinglish Sarcasm Detection")
st.write("Detect sarcasm in Hindi-English code-mixed text.")

text = st.text_area("Enter your text:")

if st.button("Detect Sarcasm"):
    if text.strip():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)[0]

        prediction = torch.argmax(probabilities).item()

        if prediction == 1:
            st.error("SARCASTIC")
        else:
            st.success("NON-SARCASTIC")

        st.write(f"Non-sarcastic: {probabilities[0].item()*100:.2f}%")
        st.write(f"Sarcastic: {probabilities[1].item()*100:.2f}%")
