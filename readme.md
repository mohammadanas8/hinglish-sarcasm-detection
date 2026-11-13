# Hinglish Sarcasm Detection

## Project Overview
An AI-based application designed to detect sarcasm in
Hindi-English (Hinglish) code-mixed social media text.

## Model
- Multilingual BERT (mBERT)
- Model: google-bert/bert-base-multilingual-cased
- Fine-tuned for binary sarcasm classification
- Maximum sequence length: 128

## Classification
- 0: Non-Sarcastic
- 1: Sarcastic

## Technologies Used
- Python
- PyTorch
- Hugging Face Transformers
- Streamlit
- Pandas
- Matplotlib
- Seaborn

## Dataset
- Total samples: 8,959
- Training: 7,167
- Validation: 896
- Testing: 896

## Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 94.98% |
| Precision | 94.43% |
| Recall | 96.54% |
| Sarcasm F1 | 95.48% |
| Macro F1 | 94.92% |
| Weighted F1 | 94.97% |

## Application Features
- Hinglish text input
- Sarcasm prediction
- Probability distribution
- Prediction confidence
- Prediction history
- CSV export
- Performance dashboard
- Confusion matrix
- Tokenization visualization

## Run Locally

Install dependencies:

pip install -r requirements.txt

Start the application:

python -m streamlit run app.py

## Model Repository

mohammadanas8/hinglish-sarcasm-mbert