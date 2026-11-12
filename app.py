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

text = st.text_area("Enter your Hinglish text:")

if st.button("Detect Sarcasm"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]

        prediction = probs.argmax().item()

        if prediction == 1:
            st.error("SARCASTIC")
        else:
            st.success("NON-SARCASTIC")

        st.write(f"Non-sarcastic: {probs[0].item()*100:.2f}%")
        st.write(f"Sarcastic: {probs[1].item()*100:.2f}%")

        st.info(
            "This model is designed primarily for Hindi-English "
            "code-mixed text. English-only predictions may be unreliable."
        )
