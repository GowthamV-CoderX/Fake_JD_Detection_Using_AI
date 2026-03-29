"""
Simple Streamlit UI for Job Scam Detector
Clean, minimal interface - just paste a job description and see if it's real or fake
Run: streamlit run app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from hybrid_detector import HybridDetector

# Page config
st.set_page_config(
    page_title="Job Detector",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Minimal styling
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

# Load detector
@st.cache_resource
def load_detector():
    detector = HybridDetector(use_embeddings=False)
    model_dir = Path("models")
    if model_dir.exists() and (model_dir / "tfidf_lr.joblib").exists():
        try:
            detector.load(str(model_dir))
            return detector, True
        except Exception:
            pass
    return detector, False

detector, model_loaded = load_detector()

# Simple title
st.markdown("# ✨ Job Detector")
st.markdown("*Is this job description real or fake?*")
st.markdown("---")

# Input
st.markdown("### 📝 Paste Job Description")
job_text = st.text_area(
    "Job Description",
    height=200,
    placeholder="Paste the job description here...",
    label_visibility="collapsed"
)

# Analyze button
if st.button("🔍 Check", type="primary", use_container_width=True):
    if job_text.strip():
        with st.spinner("Analyzing..."):
            t0 = time.perf_counter()
            result = detector.predict(text=job_text)
            elapsed = (time.perf_counter() - t0) * 1000

        # Display result
        is_fake = result.prediction == "FAKE"
        result_class = "result-fake" if is_fake else "result-real"
        verdict_text = "🚨 FAKE" if is_fake else "✅ REAL"
        
        st.markdown(f"""
        <div class="{result_class}">
            <div class="confidence">{verdict_text}</div>
            <div>Confidence: <strong>{result.confidence}%</strong></div>
            <div>Risk Level: <strong>{result.risk_level}</strong></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show details if there are risk flags
        if result.risk_flags:
            st.markdown("### ⚠️ Red Flags Detected")
            for flag in result.risk_flags:
                st.write(f"• {flag}")
        
        # Explanation
        st.markdown("### 📌 Why?")
        st.info(result.explanation)
        
        st.caption(f"Analyzed in {elapsed:.0f}ms")
    else:
        st.warning("Please paste a job description to analyze.")

st.markdown("---")
st.markdown("""
<small style="color: gray; text-align: center; display: block;">
Model Status: {'✅ Pre-trained models loaded' if model_loaded else '⚡ Using rule-based detection only'}
</small>
""", unsafe_allow_html=True)
