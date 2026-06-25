import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
import joblib

# DYNAMIC HARDCODED PATHS (Always targets the directory where this file lives)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find dataset at expected path: '{DATA_PATH}'")

    print(f"--- Loading Data from: {DATA_PATH} ---")
    df = pd.read_csv(DATA_PATH)
    
    df['text'] = df['text'].fillna('')
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    X = df['cleaned_text']
    y = df['target']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("--- Training Pipeline ---")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=2)),
        ('clf', LogisticRegression(C=2.0, max_iter=1000, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    print("--- Validation Metrics ---")
    val_preds = pipeline.predict(X_val)
    print(classification_report(y_val, val_preds))
    print(f"Validation F1-Score: {f1_score(y_val, val_preds):.4f}")
    
    print(f"--- Saving Trained Model to: {MODEL_PATH} ---")
    joblib.dump(pipeline, MODEL_PATH)
    print("Model successfully saved!")

if __name__ == "__main__":
    train_model()