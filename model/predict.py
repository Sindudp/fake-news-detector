"""
predict.py — Load saved model and make predictions
"""

import joblib
import re
import os

MODEL_DIR = "model/saved"

_model      = None
_vectorizer = None

def _load():
    global _model, _vectorizer
    if _model is None:
        _model      = joblib.load(f"{MODEL_DIR}/model.pkl")
        _vectorizer = joblib.load(f"{MODEL_DIR}/vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict(text: str) -> dict:
    """
    Returns:
        label      : 'FAKE' or 'REAL'
        confidence : float 0-100
        proba      : [prob_real, prob_fake]
    """
    _load()

    cleaned = clean_text(text)
    vec     = _vectorizer.transform([cleaned])
    proba   = _model.predict_proba(vec)[0]   # [prob_real, prob_fake]

    label      = 'FAKE' if proba[1] > 0.5 else 'REAL'
    confidence = round(float(max(proba)) * 100, 1)

    return {
        'label'     : label,
        'confidence': confidence,
        'proba'     : proba.tolist(),
    }

if __name__ == "__main__":
    sample = "Scientists discover new treatment that cures all diseases overnight"
    result = predict(sample)
    print(f"Text   : {sample}")
    print(f"Label  : {result['label']}")
    print(f"Confidence: {result['confidence']}%")