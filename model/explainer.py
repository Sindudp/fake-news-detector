"""
explainer.py — LIME explainability for word-level highlighting
"""

import joblib
import re
import numpy as np
from lime.lime_text import LimeTextExplainer

MODEL_DIR = "model/saved"

_model      = None
_vectorizer = None
_explainer  = LimeTextExplainer(class_names=['Real', 'Fake'])

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

def _predict_proba(texts):
    cleaned = [clean_text(t) for t in texts]
    vecs    = _vectorizer.transform(cleaned)
    return _model.predict_proba(vecs)

def explain(text: str, num_features: int = 15) -> list[tuple[str, float]]:
    """
    Returns list of (word, weight) tuples.
    Positive weight → pushes toward FAKE
    Negative weight → pushes toward REAL
    """
    _load()
    exp = _explainer.explain_instance(
        text,
        _predict_proba,
        num_features=num_features,
        labels=[1]   # explain the FAKE class
    )
    return exp.as_list(label=1)   # [(word, weight), ...]

def highlight_words(text: str, explanation: list) -> list[dict]:
    """
    Returns list of {word, weight, color_intensity} for UI rendering.
    """
    word_weights = {word.lower(): weight for word, weight in explanation}
    words        = text.split()
    result       = []

    for word in words:
        clean = re.sub(r'[^a-z]', '', word.lower())
        weight = word_weights.get(clean, 0.0)
        result.append({
            'word'            : word,
            'weight'          : round(weight, 4),
            'color_intensity' : min(abs(weight) * 5, 1.0),  # 0–1 scale
            'direction'       : 'fake' if weight > 0 else ('real' if weight < 0 else 'neutral')
        })

    return result