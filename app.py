"""
app.py — Fake News Detector | Streamlit UI
Run: streamlit run app.py
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.predict   import predict
from model.explainer import explain, highlight_words

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .verdict-fake { background:#fff0f0; border:1.5px solid #e74c3c;
                    border-radius:10px; padding:1rem; text-align:center; }
    .verdict-real { background:#f0fff4; border:1.5px solid #27ae60;
                    border-radius:10px; padding:1rem; text-align:center; }
    .verdict-label { font-size:2rem; font-weight:700; }
    .verdict-conf  { font-size:1rem; color:#555; margin-top:4px; }
    .word-highlight span { padding:2px 4px; border-radius:3px;
                           margin:1px; display:inline-block; font-size:15px; }
    .legend-box { display:flex; gap:1rem; font-size:13px; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────
st.title("🔍 Fake News Detector")
st.caption("Paste any news headline or article. The AI will classify it and show *why*.")

st.divider()

# ── Sample Headlines ──────────────────────────────────────
SAMPLES = {
    "Try a sample...": "",
    "🟢 Likely Real — Climate report": "Scientists warn that global temperatures have risen 1.2 degrees Celsius above pre-industrial levels according to the latest IPCC report released Monday.",
    "🔴 Likely Fake — Miracle cure": "Doctors HATE her! Local woman cures cancer overnight using one simple trick that Big Pharma doesn't want you to know about.",
    "🟢 Likely Real — Tech news": "Apple announced its quarterly earnings exceeding analyst expectations, driven by strong iPhone 15 sales in emerging markets.",
    "🔴 Likely Fake — Election": "BREAKING: Thousands of illegal ballots found in warehouse! Election officials caught destroying evidence — share before they delete this!",
}

sample_choice = st.selectbox("Or pick a sample headline:", list(SAMPLES.keys()))

# ── Text Input ────────────────────────────────────────────
user_text = st.text_area(
    "Paste your news headline or article:",
    value=SAMPLES[sample_choice],
    height=160,
    placeholder="e.g. Scientists discover new planet in solar system..."
)

analyse_btn = st.button("🔍 Analyse", type="primary", use_container_width=True)

# ── Analysis ──────────────────────────────────────────────
if analyse_btn:
    if not user_text.strip():
        st.warning("Please enter some text first.")
    elif len(user_text.split()) < 5:
        st.warning("Please enter at least 5 words for a reliable result.")
    else:
        with st.spinner("Analysing..."):
            result = predict(user_text)
            exp    = explain(user_text)
            words  = highlight_words(user_text, exp)

        st.divider()

        # ── Verdict Card ──────────────────────────────────
        label = result['label']
        conf  = result['confidence']
        css   = "verdict-fake" if label == "FAKE" else "verdict-real"
        icon  = "🚨" if label == "FAKE" else "✅"
        color = "#e74c3c" if label == "FAKE" else "#27ae60"

        st.markdown(f"""
        <div class="{css}">
            <div class="verdict-label" style="color:{color}">{icon} {label}</div>
            <div class="verdict-conf">Confidence: <strong>{conf}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # ── Confidence Bar ────────────────────────────────
        real_pct = round(result['proba'][0] * 100, 1)
        fake_pct = round(result['proba'][1] * 100, 1)

        col1, col2 = st.columns(2)
        col1.metric("Real probability", f"{real_pct}%")
        col2.metric("Fake probability", f"{fake_pct}%")

        st.progress(fake_pct / 100)

        st.divider()

        # ── Word Highlights ───────────────────────────────
        st.subheader("🔬 Why did the model decide this?")
        st.caption("Words highlighted in red pushed toward FAKE. Green pushed toward REAL.")

        html_words = ""
        for w in words:
            intensity = w['color_intensity']
            if w['direction'] == 'fake':
                r, g, b = 255, int(255 * (1 - intensity * 0.7)), int(255 * (1 - intensity * 0.7))
            elif w['direction'] == 'real':
                r, g, b = int(255 * (1 - intensity * 0.7)), 255, int(255 * (1 - intensity * 0.7))
            else:
                r, g, b = 240, 240, 240

            bg = f"rgb({r},{g},{b})"
            html_words += f'<span style="background:{bg}; padding:2px 5px; border-radius:4px; margin:2px; display:inline-block; font-size:15px;">{w["word"]}</span> '

        st.markdown(f'<div class="word-highlight">{html_words}</div>', unsafe_allow_html=True)

        st.divider()

        # ── Top Contributing Words ────────────────────────
        st.subheader("📊 Top trigger words")
        fake_words = sorted([(w, s) for w, s in exp if s > 0], key=lambda x: -x[1])[:8]
        real_words = sorted([(w, s) for w, s in exp if s < 0], key=lambda x: x[1])[:8]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔴 Pushing toward FAKE**")
            for word, score in fake_words:
                st.markdown(f"- `{word}` &nbsp; ({score:+.3f})")

        with col2:
            st.markdown("**🟢 Pushing toward REAL**")
            for word, score in real_words:
                st.markdown(f"- `{word}` &nbsp; ({score:+.3f})")

# ── Footer ────────────────────────────────────────────────
st.divider()
st.caption("Built with scikit-learn · LIME · Streamlit · Trained on ISOT Fake News Dataset")