# 🚨 Disaster Tweet Classifier

A lightweight Machine Learning system to detect real-world emergency and disaster statements within short text phrases or tweets. Built using a TF-IDF + Logistic Regression pipeline, this project delivers fast training performance and a minimal memory footprint without requiring a GPU.

---

## Project Structure
* `app.py`: Streamlit Web User Interface dashboard.
* `predict.py`: Batch prediction command-line tool (CLI).
* `train.py`: Model training and evaluation script.
* `requirements.txt`: Pinned project library dependencies.
* `runtime.txt`: Pinned Python version.

---

## Quick Start (One Command Setup)

To test the application in a clean environment, follow these steps:

### 1. Environment Setup & Installation
Clone the repository, initialize your virtual environment, and install the pinned dependencies:
```bash
git clone <your-repo-url> && cd disaster-classifier
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt


### 2. Train the Model
Ensure you have downloaded the Kaggle dataset (train.csv) and placed it directly into the project root directory. Then run:

python train.py

This will train the pipeline, output validation metrics (F1-score), and save the serialized model asset as model.joblib dynamically in the same directory.


### 3. Running Local Web UI
To launch the dashboard application on localhost:

streamlit run app.py

Note: Your terminal will automatically open your default browser to http://localhost:8501

### 4 Batch Prediction Script (CLI)
python predict.py --input test.csv --output predictions.csv
