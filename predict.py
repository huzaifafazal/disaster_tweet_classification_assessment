import os
import re
import pandas as pd
import joblib

# DYNAMIC HARDCODED PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "test.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "predictions.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Run train.py first.")
        return

    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input dataset file '{INPUT_PATH}' not found.")
        return

    print(f"Loading model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    print(f"Reading batch data from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    # Detect text column dynamically
    text_col = None
    possible_cols = ['text', 'tweet', 'Tweet', 'TEXT', 'content']
    for col in possible_cols:
        if col in df.columns:
            text_col = col
            break
            
    if text_col is None:
        text_col = df.select_dtypes(include=['object']).columns[0]
        print(f"Warning: Defaulting to column: '{text_col}'")

    print("Running batch inference...")
    cleaned_texts = df[text_col].fillna("").apply(clean_text).tolist()
    
    preds = model.predict(cleaned_texts)
    probs = model.predict_proba(cleaned_texts)
    scores = [probs[i][pred] for i, pred in enumerate(preds)]

    output_df = pd.DataFrame({
        'text': df[text_col],
        'label': preds,
        'score': scores
    })

    print(f"Saving predictions to: {OUTPUT_PATH}")
    output_df.to_csv(OUTPUT_PATH, index=False)
    print("Success!")

if __name__ == "__main__":
    main()