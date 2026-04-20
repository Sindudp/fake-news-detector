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
    return re.sub(r'\s+', ' ', text).strip()

def predict(text: str) -> dict:
    _load()
    cleaned = clean_text(text)
    vec     = _vectorizer.transform([cleaned])
    proba   = _model.predict_proba(vec)[0]
    label      = 'FAKE' if proba[1] > 0.5 else 'REAL'
    confidence = round(float(max(proba)) * 100, 1)
    return {
        'label'     : label,
        'confidence': confidence,
        'proba'     : proba.tolist(),
    }