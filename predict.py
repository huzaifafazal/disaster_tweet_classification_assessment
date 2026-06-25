import argparse
import os
import re
import pandas as pd
import joblib

# Dynamic fallback path for the model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def main():
    # Setup the required CLI arguments
    parser = argparse.ArgumentParser(description="Batch prediction CLI for disaster tweets.")
    parser.add_argument("--input", required=True, help="Path to the input CSV file (e.g., tweets.csv)")
    parser.add_argument("--output", required=True, help="Path where the output predictions CSV will be saved")
    args = parser.parse_args()

    # Validate model existence
    if not os.path.exists(DEFAULT_MODEL_PATH):
        print(f"Error: Model file '{DEFAULT_MODEL_PATH}' not found. Please run train.py first.")
        return

    # Validate input file existence
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    print(f"Loading model from: {DEFAULT_MODEL_PATH}")
    model = joblib.load(DEFAULT_MODEL_PATH)

    print(f"Reading batch data from: {args.input}")
    df = pd.read_csv(args.input)

    # Detect the text column dynamically
    text_col = None
    possible_cols = ['text', 'tweet', 'Tweet', 'TEXT', 'content']
    for col in possible_cols:
        if col in df.columns:
            text_col = col
            break
            
    if text_col is None:
        text_col = df.select_dtypes(include=['object']).columns[0]
        print(f"Warning: No explicit 'text' column found. Defaulting to column: '{text_col}'")

    print("Running batch inference...")
    cleaned_texts = df[text_col].fillna("").apply(clean_text).tolist()
    
    preds = model.predict(cleaned_texts)
    probs = model.predict_proba(cleaned_texts)
    
    # Extract confidence score for the predicted label
    scores = [probs[i][pred] for i, pred in enumerate(preds)]

    # Format output exactly as requested: text, label, score
    output_df = pd.DataFrame({
        'text': df[text_col],
        'label': preds,
        'score': scores
    })

    print(f"Saving predictions to: {args.output}")
    output_df.to_csv(args.output, index=False)
    print("Success!")

if __name__ == "__main__":
    main()