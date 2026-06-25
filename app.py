import streamlit as st
import joblib
import os
import re

# Page config
st.set_page_config(page_title="Disaster Tweet Detector", page_icon="🚨", layout="centered")

# DYNAMIC HARDCODED PATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    return re.sub(r'\s+', ' ', text).strip()

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

# App UI Header
st.title("🚨 Disaster Tweet Classifier")
st.markdown("Enter a short text or tweet below to evaluate whether it describes a real-world emergency/disaster scenario.")

model = load_model()

if model is None:
    st.error(f"⚠️ Model file not found at `{MODEL_PATH}`! Please run `train.py` first.")
else:
    user_input = st.text_area("Tweet / Text Input", placeholder="Type or paste tweet content here...", height=120)
    
    if st.button("Analyze Text", type="primary"):
        cleaned = clean_text(user_input)
        
        if not cleaned:
            st.warning("Please enter some meaningful text before hitting analyze.")
        else:
            prediction = model.predict([cleaned])[0]
            probabilities = model.predict_proba([cleaned])[0]
            confidence = probabilities[prediction]
            
            st.write("---")
            st.subheader("Analysis Result")
            
            if prediction == 1:
                st.error(f"**🚨 Real Disaster Detected** (Confidence: {confidence:.2%})")
                st.markdown("> **Note:** The model flags this text as an active emergency, accident, or natural disaster report.")
            else:
                st.success(f"**✅ Normal / Non-Disaster** (Confidence: {confidence:.2%})")
                st.markdown("> **Note:** The model identifies this text as casual communication, metaphor, or non-emergency context.")
                
            st.progress(float(probabilities[1]), text=f"Emergency Intent Probability Score: {probabilities[1]:.2%}")