import os
import re
import pickle
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import torch
from captum.attr import IntegratedGradients

# ---------------------------- Text Preprocessing ----------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    emoji_pattern = re.compile(
        "[" 
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002700-\U000027BF"  
        u"\U000024C2-\U0001F251"
        "]", flags=re.UNICODE)
    text = emoji_pattern.sub("", text)
    text = re.sub(r'[^A-Za-z0-9 ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------------- Inference Pipeline ----------------------------
class InferencePipeline:
    def __init__(self, model, tokenizer, clean_fn=clean_text, max_length: int = 128):
        self.model = model
        self.tokenizer = tokenizer
        self.clean_fn = clean_fn
        self.max_length = max_length
        self.model.eval()
        self.ig = IntegratedGradients(self.forward)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def predict_single(self, text: str) -> Dict[str, Any]:
        cleaned = self.clean_fn(text)
        enc = self.tokenizer(
            cleaned,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", None)

        with torch.no_grad():
            out = self.model(**enc)
            logits = out.logits
            pred_id = int(torch.argmax(logits, dim=-1).item())
            pred_label = {0:"Negative",1:"Neutral",2:"Positive"}.get(pred_id,"Unknown")

        # -------- XAI Explanation --------
        attr = self.ig.attribute(inputs=input_ids, target=pred_id, additional_forward_args=attention_mask)
        attr = attr.sum(dim=-1).squeeze(0)
        attr = attr.detach().cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # نختار الكلمات الاكثر تأثيراً
        top_indices = attr.argsort()[-5:][::-1]
        top_words = [tokens[i] for i in top_indices if tokens[i].isalnum()]
        
        explanation = f"{pred_label} due to influential words such as " + ", ".join([f"'{w}'" for w in top_words]) + "."

        return {"pred_id": pred_id, "pred_label": pred_label, "explanation": explanation}


# Default label mapping
DEFAULT_ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}

# ---------------------------- Load Serialized Model ----------------------------
MODEL_FILENAME = "roberta_pipeline.pkl"
pipeline = None
if os.path.exists(MODEL_FILENAME):
    try:
        with open(MODEL_FILENAME, "rb") as f:
            pipeline = pickle.load(f)
        MODEL_LOADED = True
    except Exception:
        MODEL_LOADED = False
else:
    MODEL_LOADED = False

# ---------------------------- Streamlit Setup ----------------------------
st.set_page_config(page_title="Product Reviews Sentiment Analysis", layout="wide")

st.markdown("<h1 style='text-align:center;'>Product Reviews Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Analyze single text reviews or batch CSV files.</p>", unsafe_allow_html=True)
st.markdown("---")

st.sidebar.title("Configuration")
input_mode = st.sidebar.radio("Select input mode:", ["Single Text", "Batch CSV"])

# ---------------------------- Single Text Mode ----------------------------
if input_mode == "Single Text" and pipeline:
    st.subheader("Single Review Prediction")
    text_input = st.text_area("Enter a review:", height=120)

    if st.button("🔍 Predict"):
        if not text_input.strip():
            st.warning("Please enter text before predicting.")
        else:
            try:
                result = pipeline.predict_single(text_input)
                st.write("**Prediction ID:**", result["pred_id"])
                st.write("**Predicted Label:**", result["pred_label"])
                st.write("**Explanation:**", result["explanation"])
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ---------------------------- Batch CSV Mode ----------------------------
elif input_mode == "Batch CSV" and pipeline:
    st.subheader("Batch Prediction (CSV Upload)")
    csv_file = st.file_uploader("Upload a CSV file", type=["csv"])
    text_column = "Text"
    batch_size = 64

    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file)
            st.write("### Preview")
            st.dataframe(df.head())
            if text_column not in df.columns:
                st.error(f"Column '{text_column}' not found.")
            else:
                if st.button("🚀 Run Batch Prediction"):
                    texts = df[text_column].astype(str).tolist()
                    predictions = []
                    explanations = []

                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i + batch_size]
                        for t in batch:
                            if t.strip() == "" or t.lower() == "nan":
                                predictions.append("empty")
                                explanations.append("No text provided.")
                            else:
                                pred = pipeline.predict_single(t)
                                predictions.append(pred["pred_label"])
                                explanations.append(pred["explanation"])

                    df["pred_label"] = predictions
                    df["explanation"] = explanations
                    st.success("Batch prediction completed successfully.")
                    st.dataframe(df.head(10))

                    csv_data = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Results CSV", data=csv_data, file_name="sentiment_predictions.csv")

        except Exception as e:
            st.error(f"Error processing CSV: {e}")

st.markdown("---")
st.caption("💡 This dashboard predicts sentiment and explains influential words for product reviews.")
