"""
download_data.py — Instructions + helper to get the ISOT dataset

The ISOT dataset is hosted on Kaggle. Two ways to get it:

── Option A: Manual (easiest) ────────────────────────────────────────────
1. Go to: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. Click Download
3. Unzip and place True.csv and Fake.csv inside the data/ folder
4. Then run this script to merge them into dataset.csv

── Option B: Kaggle API (recommended) ───────────────────────────────────
1. Install: pip install kaggle
2. Get your API key from https://www.kaggle.com/account → Create New API Token
3. Place kaggle.json in ~/.kaggle/
4. Run: kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p data/ --unzip
5. Then run this script to merge

"""

import pandas as pd
import os

DATA_DIR = "data"

def merge_isot():
    true_path = os.path.join(DATA_DIR, "True.csv")
    fake_path = os.path.join(DATA_DIR, "Fake.csv")

    if not os.path.exists(true_path) or not os.path.exists(fake_path):
        print("❌ True.csv or Fake.csv not found in data/ folder.")
        print("   Please download the ISOT dataset from Kaggle first.")
        print("   See instructions at the top of this file.")
        return

    print("📂 Loading True.csv and Fake.csv...")
    df_real = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    df_real['label'] = 0   # 0 = Real
    df_fake['label'] = 1   # 1 = Fake

    df = pd.concat([df_real, df_fake], ignore_index=True)
    df = df[['title', 'text', 'label']]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    out_path = os.path.join(DATA_DIR, "dataset.csv")
    df.to_csv(out_path, index=False)

    print(f"✅ Merged dataset saved to {out_path}")
    print(f"   Total articles : {len(df)}")
    print(f"   Real           : {(df['label']==0).sum()}")
    print(f"   Fake           : {(df['label']==1).sum()}")

if __name__ == "__main__":
    merge_isot()