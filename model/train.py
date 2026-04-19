"""
train.py — Train and compare multiple models
Run: python model/train.py
"""

import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import re
import nltk
nltk.download('stopwords', quiet=True)

DATA_PATH = "data/dataset.csv"
MODEL_DIR = "model/saved"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    print("📂 Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df['content'] = df['title'].fillna('') + " " + df['text'].fillna('')
    df = df[['content', 'label']].dropna()
    print(f"   ✅ {len(df)} articles | Real: {(df['label']==0).sum()} | Fake: {(df['label']==1).sum()}")
    return df

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def train():
    df = load_data()
    print("\n🧹 Cleaning text...")
    df['content'] = df['content'].apply(clean_text)

    X, y = df['content'], df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n🔢 Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2),
                                  stop_words='english', min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
        "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost"            : XGBClassifier(n_estimators=100, random_state=42,
                                             eval_metric='logloss', verbosity=0),
    }

    results  = {}
    best_acc, best_model, best_name = 0, None, ""

    print(f"\n🤖 Training all models...\n")
    print(f"{'Model':<25} {'Accuracy':>10} {'Time':>8}")
    print("─" * 45)

    for name, model in models.items():
        start = time.time()
        model.fit(X_train_vec, y_train)
        elapsed = time.time() - start
        acc = accuracy_score(y_test, model.predict(X_test_vec))
        results[name] = round(acc * 100, 2)
        print(f"{name:<25} {acc*100:>9.2f}%  {elapsed:>6.1f}s")
        if acc > best_acc:
            best_acc, best_model, best_name = acc, model, name

    print(f"\n🏆 Best model: {best_name} ({best_acc*100:.2f}%)")
    joblib.dump(best_model, f"{MODEL_DIR}/model.pkl")
    joblib.dump(vectorizer, f"{MODEL_DIR}/vectorizer.pkl")
    pd.DataFrame(list(results.items()), columns=['Model','Accuracy']).to_csv(
        f"{MODEL_DIR}/model_comparison.csv", index=False)
    print(f"💾 Saved to {MODEL_DIR}/")
    print(classification_report(y_test, best_model.predict(X_test_vec), target_names=['Real','Fake']))

if __name__ == "__main__":
    train()