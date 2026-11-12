import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# =====================================================
# 1. CONFIGURATION
# =====================================================

MODEL_ID = "mohammadanas8/hinglish-sarcasm-mbert"
MAX_LENGTH = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

LABELS = {
    0: "Non-Sarcastic",
    1: "Sarcastic"
}

# Initialize session history
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# =====================================================
# 2. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Hinglish Sarcasm Detector",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 3. CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #0e1117;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    color: white;
}

.subtitle {
    text-align: center;
    color: #aab2c0;
    font-size: 17px;
    margin-bottom: 30px;
}

.result-box {
    padding: 22px;
    border-radius: 12px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    margin-top: 20px;
}

.sarcastic {
    background-color: #421c25;
    color: #ff7777;
    border: 1px solid #823442;
}

.non-sarcastic {
    background-color: #10352b;
    color: #43d9a3;
    border: 1px solid #26785c;
}

.footer {
    text-align: center;
    color: #777f8d;
    padding: 25px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 4. LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# =====================================================
# 5. HEADER
# =====================================================

st.markdown(
    '<div class="main-title">Hinglish Sarcasm Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered sarcasm detection for Hindi-English '
    'code-mixed social media text.'
    '</div>',
    unsafe_allow_html=True
)


# =====================================================
# 6. SIDEBAR
# =====================================================

with st.sidebar:

    st.title("About the Project")

    st.write(
        "This application uses Multilingual BERT (mBERT) "
        "to identify sarcasm in Hindi-English text."
    )

    st.divider()

    st.subheader("Classification")

    st.success("0 — Non-Sarcastic")
    st.error("1 — Sarcastic")

    st.divider()

    st.subheader("Model Information")

    st.write("**Model:** mBERT")
    st.write("**Architecture:** BERT")
    st.write("**Maximum Length:** 128")
    st.write("**Number of Classes:** 2")

    st.write(f"**Device:** {DEVICE}")

    st.divider()

    st.caption(
        "Hinglish Sarcasm Detection "
        "| Academic AI/ML Project"
    )


# =====================================================
# 7. LOAD MODEL STATUS
# =====================================================

try:

    with st.spinner("Loading mBERT model..."):

        tokenizer, model = load_model()

    st.success(
        f"mBERT loaded successfully on {DEVICE}."
    )

except Exception as e:

    st.error(f"Model loading failed: {e}")
    st.stop()


# =====================================================
# 8. EXAMPLE SENTENCES
# =====================================================

st.header("Try Example Sentences")

examples = {

    "Select an example": "",

    "Sarcastic Hinglish":
        "Wah bhai, kya zabardast idea diya hai!",

    "Non-Sarcastic Hinglish":
        "Tumne bahut achha kaam kiya, thank you!",

    "Sarcastic English":
        "I absolutely love waiting for three hours.",

    "Non-Sarcastic English":
        "Thank you for helping me."

}

selected_example = st.selectbox(
    "Choose an example:",
    list(examples.keys())
)


# =====================================================
# 9. USER INPUT
# =====================================================

st.header("Enter Your Text")

text = st.text_area(
    "Enter a Hindi-English sentence:",
    value=examples[selected_example],
    height=130,
    placeholder="Example: Pagal hai kya, sojao raat hogayi hai."
)


# =====================================================
# 10. PREDICTION
# =====================================================

if st.button(
    "Detect Sarcasm",
    type="primary",
    use_container_width=True
):

    if not text.strip():

        st.warning("Please enter a sentence first.")

    else:

        try:

            with st.spinner("Analysing your sentence..."):

                # Tokenization using raw input text
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=MAX_LENGTH
                )

                inputs = {
                    key: value.to(DEVICE)
                    for key, value in inputs.items()
                }

                # Model inference
                with torch.no_grad():

                    outputs = model(**inputs)

                    probabilities = torch.softmax(
                        outputs.logits,
                        dim=1
                    )[0]

                prediction = torch.argmax(
                    probabilities
                ).item()

                confidence = (
                    probabilities[prediction].item() * 100
                )

                non_sarcastic = (
                    probabilities[0].item() * 100
                )

                sarcastic = (
                    probabilities[1].item() * 100
                )

            # Save prediction history
            st.session_state.prediction_history.append({

                "Text": text,

                "Prediction": LABELS[prediction],

                "Confidence": f"{confidence:.2f}%"

            })

            # =========================================
            # 11. DISPLAY PREDICTION
            # =========================================

            st.divider()

            st.header("Prediction Result")

            if prediction == 1:

                st.markdown(
                    '<div class="result-box sarcastic">'
                    'SARCASTIC'
                    '</div>',
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    '<div class="result-box non-sarcastic">'
                    'NON-SARCASTIC'
                    '</div>',
                    unsafe_allow_html=True
                )

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            # =========================================
            # 12. PROBABILITY DISTRIBUTION
            # =========================================

            st.subheader("Class Probabilities")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Non-Sarcastic",
                    f"{non_sarcastic:.2f}%"
                )

            with col2:

                st.metric(
                    "Sarcastic",
                    f"{sarcastic:.2f}%"
                )

            st.write("Probability Distribution")

            st.progress(
                int(round(sarcastic)),
                text=f"Sarcastic: {sarcastic:.2f}%"
            )

            st.progress(
                int(round(non_sarcastic)),
                text=f"Non-Sarcastic: {non_sarcastic:.2f}%"
            )

            # =========================================
            # 13. INTERPRETATION
            # =========================================

            st.subheader("Interpretation")

            if confidence >= 90:

                st.info(
                    "The model assigns a high probability "
                    "to its predicted class. This does not "
                    "guarantee that the prediction is correct."
                )

            elif confidence >= 60:

                st.warning(
                    "The model has moderate prediction confidence. "
                    "Additional conversational context may help."
                )

            else:

                st.warning(
                    "The prediction is uncertain because "
                    "the class probabilities are relatively close."
                )

            st.caption(
                "Sarcasm is context-dependent. "
                "An isolated sentence may not reveal "
                "the speaker's actual intention."
            )

            # =========================================
            # 14. TOKENIZATION DETAILS
            # =========================================

            with st.expander("View Tokenization Details"):

                st.write("Original Input")
                st.code(text)

                st.write("Tokens")

                st.write(tokenizer.tokenize(text))

                st.write("Input IDs")

                st.write(
                    inputs["input_ids"][0].tolist()
                )

                st.write("Attention Mask")

                st.write(
                    inputs["attention_mask"][0].tolist()
                )

                st.write(
                    f"Maximum sequence length: {MAX_LENGTH}"
                )

        except Exception as e:

            st.error(f"Prediction failed: {e}")


# =====================================================
# 15. PERFORMANCE DASHBOARD
# =====================================================

st.divider()

st.header("Model Performance Dashboard")

st.write(
    "mBERT evaluation results on the held-out test dataset."
)

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", "94.98%")
col2.metric("Precision", "94.43%")
col3.metric("Recall", "96.54%")

col4, col5, col6 = st.columns(3)

col4.metric("Sarcasm F1", "95.48%")
col5.metric("Macro F1", "94.92%")
col6.metric("Weighted F1", "94.97%")


# =====================================================
# 16. CONFUSION MATRIX
# =====================================================

st.subheader("Confusion Matrix")

cm = [
    [376, 28],
    [17, 475]
]

cm_df = pd.DataFrame(
    cm,
    index=["Actual Non-Sarcastic", "Actual Sarcastic"],
    columns=["Predicted Non-Sarcastic", "Predicted Sarcastic"]
)

fig, ax = plt.subplots(figsize=(8, 5))

sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="Blues",
    linewidths=1,
    cbar=True,
    ax=ax
)

ax.set_title("mBERT Confusion Matrix")
ax.set_xlabel("Predicted Labels")
ax.set_ylabel("Actual Labels")

st.pyplot(fig)

plt.close(fig)

st.caption(
    "Rows represent actual labels and columns represent predicted labels."
)

with st.expander("Confusion Matrix Interpretation"):

    st.write("**376:** Correctly predicted non-sarcastic.")

    st.write("**28:** Non-sarcastic predicted as sarcastic.")

    st.write("**17:** Sarcastic predicted as non-sarcastic.")

    st.write("**475:** Correctly predicted sarcastic.")


# =====================================================
# 17. PREDICTION HISTORY
# =====================================================

# =====================================================
# 17. PREDICTION HISTORY
# =====================================================

st.divider()

st.header("Prediction History")

if st.session_state.prediction_history:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

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

        if st.button(
            "Clear Prediction History",
            use_container_width=True
        ):

            st.session_state.prediction_history = []
            st.rerun()

else:

    st.info(
        "No predictions yet. Analyse a sentence "
        "to see your prediction history."
    )


# =====================================================
# 18. FOOTER
# =====================================================

st.divider()

st.markdown(
    '<div class="footer">'
    'Hinglish Sarcasm Detection'
    '<br>Powered by Multilingual BERT (mBERT)'
    '<br>Academic AI/ML Project'
    '</div>',
    unsafe_allow_html=True
)
