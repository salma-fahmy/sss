import os
import re
import pickle
from io import BytesIO
import requests
import streamlit as st
import pandas as pd
import torch
import google.generativeai as genai


# ---------------------------- Configure Gemini API ----------------------------
# Use your API key directly
GEMINI_API_KEY = "AIzaSyCnF-UaGJFoDLV8ANieBcfbePLUFmJv-yM"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def summarize_review_with_gemini(cleaned_text: str) -> str:
    """
    Use Gemini API to summarize the review after cleaning.
    """
    if not GEMINI_API_KEY:
        return "⚠️ Gemini API key not configured."
    
    try:
        # List available models and filter for free-tier friendly ones
        available_models = []
        try:
            for model in genai.list_models():
                if 'generateContent' in model.supported_generation_methods:
                    # Skip experimental models (they have very limited quotas)
                    if '-exp' not in model.name and 'experimental' not in model.name.lower():
                        available_models.append(model.name)
        except Exception:
            pass
        
        # Prioritize free-tier friendly models
        preferred_models = ['models/gemini-1.5-flash', 'models/gemini-1.5-flash-8b', 'models/gemini-1.5-pro']
        
        model_to_use = None
        
        # First, try preferred models if they're available
        for preferred in preferred_models:
            if preferred in available_models:
                model_to_use = preferred
                break
        
        # If no preferred model found, use first available non-experimental model
        if not model_to_use and available_models:
            model_to_use = available_models[0]
        
        # Fallback to trying common names
        if not model_to_use:
            for model_name in ['models/gemini-1.5-flash', 'models/gemini-1.5-flash-8b', 'models/gemini-pro']:
                try:
                    model = genai.GenerativeModel(model_name)
                    model_to_use = model_name
                    break
                except Exception:
                    continue
        
        if not model_to_use:
            return "⚠️ Could not find any compatible Gemini model."
        
        model = genai.GenerativeModel(model_to_use)
        
        prompt = f"""You are a helpful assistant that summarizes customer reviews concisely.

Review: {cleaned_text}

Provide a brief summary explaining what the customer thinks, focusing on the main points (e.g., taste, price, quality, service).
Keep it short and natural. Example format: "The customer is disappointed because of taste and overprice."

Summary:"""
        
        response = model.generate_content(prompt)
        return response.text.strip()
    
    except Exception as e:
        return f"⚠️ Error generating summary: {e}"


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
        "]+", flags=re.UNICODE)
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

    def predict_single(self, text):
        cleaned = self.clean_fn(text)
        enc = self.tokenizer(
            cleaned,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        with torch.no_grad():
            out = self.model(**enc)
            logits = out.logits
            pred_id = int(torch.argmax(logits, dim=-1).item())

        return {"pred_id": pred_id}


# ---------------- Label Mapping ----------------
DEFAULT_ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


# ---------------- Load Model from Dropbox ----------------
DROPBOX_URL = "https://www.dropbox.com/scl/fi/4r5mrc3tcrthzvstjpwjn/roberta_pipeline.pkl?rlkey=i5vli1htkljftqqcou8myu8y5&st=xyk2aahu&dl=1"


@st.cache_resource
def load_pipeline():
    placeholder = st.empty()
    placeholder.info("Downloading model from Dropbox…")

    try:
        response = requests.get(DROPBOX_URL)
        response.raise_for_status()

        file_bytes = BytesIO(response.content)
        pipeline = pickle.load(file_bytes)

        placeholder.empty()
        return pipeline

    except Exception as e:
        placeholder.error(f"❌ Failed to load model: {e}")
        return None


pipeline = load_pipeline()


# ---------------------------- Streamlit Page Setup ----------------------------
st.set_page_config(page_title="Product Reviews Sentiment Analysis", layout="wide")

page_bg = """
<style>
.stApp {
    background: url('https://i.pinimg.com/736x/8a/4c/31/8a4c3184c5ae66e9f090f49db6bd445a.jpg');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
h1, h2, h3, h4 { color: #111827 !important; }
[data-testid="stSidebar"] {
    background-color: rgba(255,255,255,0.5) !important;
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)


# ---------------------------- Header ----------------------------
st.markdown("<h1 style='text-align:center;'>Product Reviews Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#111;'>Analyze single text or batch CSV files.</p>", unsafe_allow_html=True)
st.markdown("---")
