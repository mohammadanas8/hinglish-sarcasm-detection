import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="Hinglish Sarcasm Detector",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_ID = "mohammadanas8/hinglish-sarcasm-mbert"
MAX_LENGTH = 128
SARCASM_THRESHOLD = 0.40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = {0: "Non-Sarcastic", 1: "Sarcastic"}

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

st.markdown("""
<style>
.stApp { background-color: #0e1117; }
.main-title { font-size: 42px; font-weight: 800; text-align: center; color: white; }
.subtitle { text-align: center; color: #aab2c0; font-size: 17px; margin-bottom: 30px; }
.result-box { padding: 22px; border-radius: 12px; text-align: center; font-size: 26px; font-weight: bold; margin-top: 20px; }
.sarcastic { background-color: #421c25; color: #ff7777; border: 1px solid #823442; }
.non-sarcastic { background-color: #10352b; color: #43d9a3; border: 1px solid #26785c; }
.footer { text-align: center; color: #777f8d; padding: 25px; font-size: 13px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    # Fetch the latest model files from the updated Hugging Face repository.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, force_download=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, force_download=True
    )
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


st.markdown(
    '<div class="main-title">Hinglish Sarcasm Detector</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">AI-powered sarcasm detection for Hindi-English '
    'code-mixed social media text.</div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.title("About the Project")
    st.write(
        "This application uses an augmented Multilingual BERT (mBERT) "
        "model to identify sarcasm in Hindi-English text."
    )
    st.divider()
    st.subheader("Classification")
    st.success("0 — Non-Sarcastic")
    st.error("1 — Sarcastic")
    st.divider()
    st.subheader("Model Information")
    st.write("**Model:** Augmented mBERT")
    st.write("**Architecture:** Multilingual BERT")
    st.write(f"**Maximum Length:** {MAX_LENGTH}")
    st.write("**Number of Classes:** 2")
    st.write(f"**Sarcasm threshold:** {SARCASM_THRESHOLD:.2f}")
    st.write(f"**Device:** {DEVICE}")
    st.divider()
    st.caption("Hinglish Sarcasm Detection | Academic AI/ML Project")

try:
    with st.spinner("Loading updated mBERT model from Hugging Face..."):
        tokenizer, model = load_model()
    st.success(f"Updated augmented mBERT loaded successfully on {DEVICE}.")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.info(
        "Check that the Hugging Face repository is public, or configure "
        "a Hugging Face read token in Streamlit Secrets."
    )
    st.stop()

st.header("Try Example Sentences")
examples = {
    "Select an example": "",
    "Sarcastic Hinglish": "Waah, kya zabardast idea diya hai, ab sabka kaam badh gaya!",
    "Non-Sarcastic Hinglish": "Tumne bahut achha kaam kiya, thank you!",
    "Sarcastic English": "I absolutely love waiting for three hours.",
    "Non-Sarcastic English": "Thank you for helping me."
}
selected_example = st.selectbox("Choose an example:", list(examples.keys()))

st.header("Enter Your Text")
text = st.text_area(
    "Enter a Hindi-English sentence:",
    value=examples[selected_example],
    height=130,
    placeholder="Example: Pagal hai kya, sojao raat hogayi hai."
)

if st.button("Detect Sarcasm", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please enter a sentence first.")
    else:
        try:
            with st.spinner("Analysing your sentence..."):
                # Raw text is used, matching the training pipeline.
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=MAX_LENGTH
                )
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

                with torch.inference_mode():
                    outputs = model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=1)[0]

                non_sarcastic_probability = probabilities[0].item()
                sarcastic_probability = probabilities[1].item()

                # Threshold selected using validation data.
                prediction = int(sarcastic_probability >= SARCASM_THRESHOLD)
                confidence = (
                    sarcastic_probability if prediction == 1
                    else non_sarcastic_probability
                ) * 100

                non_sarcastic = non_sarcastic_probability * 100
                sarcastic = sarcastic_probability * 100

            st.session_state.prediction_history.append({
                "Text": text,
                "Prediction": LABELS[prediction],
                "Non-Sarcastic Probability": f"{non_sarcastic:.2f}%",
                "Sarcastic Probability": f"{sarcastic:.2f}%",
                "Decision Threshold": f"{SARCASM_THRESHOLD:.2f}"
            })

            st.divider()
            st.header("Prediction Result")
            if prediction == 1:
                st.markdown(
                    '<div class="result-box sarcastic">SARCASTIC</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="result-box non-sarcastic">NON-SARCASTIC</div>',
                    unsafe_allow_html=True
                )

            st.metric("Selected-class probability", f"{confidence:.2f}%")

            st.subheader("Class Probabilities")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Non-Sarcastic", f"{non_sarcastic:.2f}%")
            with col2:
                st.metric("Sarcastic", f"{sarcastic:.2f}%")

            st.progress(
                int(round(sarcastic)),
                text=f"Sarcastic probability: {sarcastic:.2f}%"
            )
            st.progress(
                int(round(non_sarcastic)),
                text=f"Non-sarcastic probability: {non_sarcastic:.2f}%"
            )

            st.caption(
                f"Decision rule: classified as sarcastic when sarcasm "
                f"probability is at least {SARCASM_THRESHOLD:.2f}. "
                "Probabilities are model outputs, not guarantees of correctness."
            )

            st.subheader("Interpretation")
            if abs(sarcastic_probability - non_sarcastic_probability) < 0.15:
                st.warning(
                    "The class probabilities are relatively close. "
                    "Additional conversational context may help."
                )
            elif confidence >= 90:
                st.info(
                    "The model assigns a high probability to its selected class. "
                    "This does not guarantee that the prediction is correct."
                )
            else:
                st.info(
                    "The model selected this class using the configured "
                    "decision threshold. Sarcasm can depend on context and tone."
                )

            st.caption(
                "Sarcasm is context-dependent. An isolated sentence may not "
                "reveal the speaker's actual intention."
            )

            with st.expander("View Tokenization Details"):
                st.write("Original Input")
                st.code(text)
                st.write("Tokens")
                st.write(tokenizer.tokenize(text))
                st.write("Input IDs")
                st.write(inputs["input_ids"][0].detach().cpu().tolist())
                st.write("Attention Mask")
                st.write(inputs["attention_mask"][0].detach().cpu().tolist())
                st.write(f"Maximum sequence length: {MAX_LENGTH}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.divider()
st.header("Model Performance Dashboard")
st.write(
    "Augmented mBERT results on the original held-out test dataset. "
    "These metrics and confusion-matrix counts were recorded using the "
    "standard 0.50 argmax decision rule. The deployed application uses "
    "the 0.40 threshold selected separately on validation data."
)

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", "95.20%")
col2.metric("Sarcasm Precision", "96.29%")
col3.metric("Sarcasm Recall", "94.92%")

col4, col5, col6 = st.columns(3)
col4.metric("Sarcasm F1", "95.60%")
col5.metric("Macro F1", "95.16%")
col6.metric("Weighted F1", "95.20%")

st.subheader("Confusion Matrix — Test Set (0.50 argmax rule)")
cm = [[386, 18], [25, 467]]
cm_df = pd.DataFrame(
    cm,
    index=["Actual Non-Sarcastic", "Actual Sarcastic"],
    columns=["Predicted Non-Sarcastic", "Predicted Sarcastic"]
)

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(
    cm_df, annot=True, fmt="d", cmap="Blues",
    linewidths=1, cbar=True, ax=ax
)
ax.set_title("Augmented mBERT Confusion Matrix")
ax.set_xlabel("Predicted Labels")
ax.set_ylabel("Actual Labels")
st.pyplot(fig)
plt.close(fig)

st.caption(
    "Rows represent actual labels and columns represent predicted labels. "
    "These counts correspond to the reported test evaluation at the "
    "standard 0.50 argmax rule."
)

with st.expander("Confusion Matrix Interpretation"):
    st.write("**386:** Correctly predicted non-sarcastic.")
    st.write("**18:** Non-sarcastic predicted as sarcastic.")
    st.write("**25:** Sarcastic predicted as non-sarcastic.")
    st.write("**467:** Correctly predicted sarcastic.")

st.divider()
st.header("Prediction History")

if st.session_state.prediction_history:
    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        csv_data = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Prediction History",
            data=csv_data,
            file_name="hinglish_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col2:
        if st.button("Clear Prediction History", use_container_width=True):
            st.session_state.prediction_history = []
            st.rerun()
else:
    st.info("No predictions yet. Analyse a sentence to see your prediction history.")

st.divider()
st.markdown(
    '<div class="footer">Hinglish Sarcasm Detection'
    '<br>Powered by Augmented Multilingual BERT (mBERT)'
    '<br>Academic AI/ML Project</div>',
    unsafe_allow_html=True
)