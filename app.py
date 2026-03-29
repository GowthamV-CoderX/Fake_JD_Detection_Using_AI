"""
app.py
======
Location in your repo: app.py  (root)

Run locally:  streamlit run app.py
Deploy:       Works on Vercel (no torch, no heavy deps)

WHAT CHANGED vs your original:
  - Reads HF_API_TOKEN env var to decide whether semantic layer is active
  - Shows clear status indicators for ML model + HF API availability
  - Added optional title/company fields to match the API
  - Removed use_embeddings=False hardcode — now auto-detected from env var
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from hybrid_detector import HybridDetector

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Job Scam Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
.result-real {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border: 2px solid #28a745;
    color: #155724;
    padding: 20px;
    border-radius: 8px;
    text-align: center;
}
.result-fake {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    border: 2px solid #dc3545;
    color: #721c24;
    padding: 20px;
    border-radius: 8px;
    text-align: center;
}
.confidence { font-size: 2em; font-weight: bold; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)


# ── Load detector (cached so it stays alive across requests) ──────────────────
@st.cache_resource
def load_detector():
    hf_token_present = bool(os.environ.get("HF_API_TOKEN", "").strip())
    detector = HybridDetector(use_embeddings=hf_token_present)

    model_dir = Path("models")
    model_loaded = False
    if model_dir.exists() and (model_dir / "tfidf.joblib").exists():
        try:
            detector.load(str(model_dir))
            model_loaded = True
        except Exception as exc:
            st.warning(f"Could not load saved models: {exc}")

    return detector, model_loaded, hf_token_present


detector, model_loaded, hf_active = load_detector()

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("# 🔍 Job Scam Detector")
st.markdown("*Paste any job description to check if it's real or fake.*")

# Status badges
col1, col2 = st.columns(2)
with col1:
    if model_loaded:
        st.success("✅ ML models loaded")
    else:
        st.info("⚡ Rule-based mode (no saved models found)")
with col2:
    if hf_active:
        st.success("✅ Semantic layer active (HF API)")
    else:
        st.warning("⚠️ Set HF_API_TOKEN for semantic layer")

st.markdown("---")

# Input fields
st.markdown("### 📝 Paste Job Description")
job_text = st.text_area(
    "Job Description",
    height=220,
    placeholder="Paste the full job description here...",
    label_visibility="collapsed",
)

col_a, col_b = st.columns(2)
with col_a:
    title = st.text_input("Job title (optional)", placeholder="e.g. Data Entry Operator")
with col_b:
    company = st.text_input("Company name (optional)", placeholder="e.g. TechCorp Pvt Ltd")

# Analyze button
if st.button("🔍 Analyze Job Post", type="primary", use_container_width=True):
    if job_text.strip():
        with st.spinner("Analyzing..."):
            t0 = time.perf_counter()
            result = detector.predict(text=job_text, title=title, company=company)
            elapsed = (time.perf_counter() - t0) * 1000

        is_fake      = result.prediction == "FAKE"
        result_class = "result-fake" if is_fake else "result-real"
        verdict_text = "🚨 FAKE JOB POST" if is_fake else "✅ LOOKS LEGITIMATE"

        st.markdown(f"""
        <div class="{result_class}">
            <div class="confidence">{verdict_text}</div>
            <div>Confidence: <strong>{result.confidence}%</strong></div>
            <div>Risk Level: <strong>{result.risk_level}</strong></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")  # spacer

        if result.risk_flags:
            st.markdown("### ⚠️ Red Flags Detected")
            for flag in result.risk_flags:
                st.write(f"• {flag}")

        st.markdown("### 📌 Explanation")
        st.info(result.explanation)

        with st.expander("🔢 Detailed scores"):
            st.json(result.scores)

        st.caption(f"Analyzed in {elapsed:.0f} ms")
    else:
        st.warning("Please paste a job description first.")

st.markdown("---")
st.markdown(
    "<small style='color:gray;text-align:center;display:block;'>"
    "Detection layers: Rule engine (15 rules) + TF-IDF + Semantic embeddings via HF API"
    "</small>",
    unsafe_allow_html=True,
)