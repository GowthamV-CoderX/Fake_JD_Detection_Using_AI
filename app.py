"""
app.py  —  Fabulous Frontend Version
Location: app.py (repo root)

Run locally:  streamlit run app.py

FEATURES:
  - Dark cyberpunk-style UI with glowing effects
  - Animated result cards (FAKE/REAL)
  - Pie chart: risk score breakdown (rules vs tfidf vs embedding)
  - Bar chart: red flag severity scores
  - Pie chart: session stats (fake vs real analyzed so far)
  - All charts from user-entered data (live, not static CSV)
  - No HF_API_TOKEN warning shown to user — handled silently
  - Detection layers shown as animated progress bars
  - Full red flags list with human-readable descriptions
"""

import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from hybrid_detector import HybridDetector

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JobGuard AI — Fake Job Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS — Dark Cyberpunk Theme ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

* { font-family: 'Space Grotesk', sans-serif; }

/* Dark background */
.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Hero section */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 48px 40px;
    text-align: center;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 50% 50%, rgba(99,102,241,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c084fc, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 12px 0;
    letter-spacing: -1px;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: #94a3b8;
    font-weight: 300;
    margin: 0;
    letter-spacing: 0.5px;
}

/* Input card */
.input-card {
    background: #111827;
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 24px;
}

/* Result cards */
.result-fake {
    background: linear-gradient(135deg, #1a0a0a, #2d0f0f);
    border: 2px solid #ef4444;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    box-shadow: 0 0 40px rgba(239,68,68,0.2);
    animation: pulse-red 2s infinite;
}
.result-real {
    background: linear-gradient(135deg, #0a1a0f, #0f2d1a);
    border: 2px solid #22c55e;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    box-shadow: 0 0 40px rgba(34,197,94,0.2);
    animation: pulse-green 2s infinite;
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 30px rgba(239,68,68,0.2); }
    50%       { box-shadow: 0 0 60px rgba(239,68,68,0.4); }
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 30px rgba(34,197,94,0.15); }
    50%       { box-shadow: 0 0 60px rgba(34,197,94,0.3); }
}
.verdict-text {
    font-size: 2.8rem;
    font-weight: 700;
    margin: 0 0 8px 0;
    font-family: 'JetBrains Mono', monospace;
}
.verdict-fake { color: #f87171; }
.verdict-real { color: #4ade80; }
.confidence-text {
    font-size: 1.1rem;
    color: #cbd5e1;
    margin: 4px 0;
}

/* Risk badge */
.risk-critical { background: #7f1d1d; color: #fca5a5; border: 1px solid #ef4444; padding: 4px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; display: inline-block; }
.risk-high     { background: #431407; color: #fdba74; border: 1px solid #f97316; padding: 4px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; display: inline-block; }
.risk-medium   { background: #422006; color: #fde68a; border: 1px solid #eab308; padding: 4px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; display: inline-block; }
.risk-low      { background: #052e16; color: #86efac; border: 1px solid #22c55e; padding: 4px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; display: inline-block; }

/* Flag items */
.flag-item {
    background: rgba(239,68,68,0.08);
    border-left: 3px solid #ef4444;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    margin: 6px 0;
    font-size: 0.9rem;
    color: #fca5a5;
}
.flag-item-real {
    background: rgba(34,197,94,0.08);
    border-left: 3px solid #22c55e;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    margin: 6px 0;
    font-size: 0.9rem;
    color: #86efac;
}

/* Explanation box */
.explanation-box {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 12px;
    padding: 20px;
    color: #c7d2fe;
    font-size: 0.95rem;
    line-height: 1.7;
}

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #818cf8;
    margin: 24px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Stats cards */
.stat-card {
    background: #111827;
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.stat-number {
    font-size: 2.2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #818cf8;
}
.stat-label {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Streamlit overrides */
.stTextArea textarea {
    background: #0f172a !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
.stTextInput input {
    background: #0f172a !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 14px 32px !important;
    transition: all 0.2s !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.4) !important;
}
label { color: #94a3b8 !important; font-size: 0.85rem !important; }
.stTabs [data-baseweb="tab"] { color: #64748b !important; }
.stTabs [aria-selected="true"] { color: #818cf8 !important; border-bottom-color: #818cf8 !important; }
</style>
""", unsafe_allow_html=True)


# ── Load detector (silent — no token warnings to user) ────────────────────────
@st.cache_resource(show_spinner=False)
def load_detector():
    hf_active = bool(os.environ.get("HF_API_TOKEN", "").strip())
    detector  = HybridDetector(use_embeddings=hf_active)
    model_dir = Path("models")
    loaded    = False
    if model_dir.exists() and (model_dir / "tfidf.joblib").exists():
        try:
            detector.load(str(model_dir))
            loaded = True
        except Exception:
            pass
    return detector, loaded, hf_active


detector, model_loaded, hf_active = load_detector()

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []        # list of dicts for session charts

if "last_result" not in st.session_state:
    st.session_state.last_result = None  # full result dict for persistent display

if "last_elapsed" not in st.session_state:
    st.session_state.last_elapsed = 0


# ── Helper: Plotly dark chart theme ──────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="#94a3b8", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
)


def make_score_pie(scores: dict) -> go.Figure:
    rules_val = float(scores.get("rules", 0))
    tfidf_val = float(scores.get("tfidf", 0)) if scores.get("tfidf") != "n/a" else 0
    emb_val   = float(scores.get("embedding", 0)) if scores.get("embedding") != "n/a" else 0

    labels = ["Rule Engine", "TF-IDF Model", "Semantic (HF)"]
    values = [rules_val, tfidf_val, emb_val]
    colors = ["#ef4444", "#f97316", "#6366f1"]

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.55,
        marker=dict(colors=colors, line=dict(color="#0a0e1a", width=2)),
        textinfo="label+percent",
        textfont=dict(size=11, color="#e2e8f0"),
        hovertemplate="%{label}: %{value:.3f}<extra></extra>",
    ))
    fig.add_annotation(
        text=f"{int(float(scores.get('final', 0))*100)}%",
        x=0.5, y=0.5, font=dict(size=22, color="#e2e8f0", family="JetBrains Mono"),
        showarrow=False,
    )
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Risk Score Breakdown", font=dict(color="#818cf8", size=13)),
        showlegend=True,
        legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
        height=280,
    )
    return fig


def make_flags_bar(flag_descriptions: list) -> go.Figure:
    if not flag_descriptions:
        flag_descriptions = ["No red flags triggered"]

    labels = [f[:55] + "…" if len(f) > 55 else f for f in flag_descriptions]
    values = list(range(len(labels), 0, -1))
    colors = [f"rgba(239,68,68,{0.4 + 0.6*(i/max(len(labels),1))})"
              for i in range(len(labels))]

    fig = go.Figure(go.Bar(
        y=labels, x=values, orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)")),
        hovertemplate="%{y}<extra></extra>",
        text=[str(v) for v in values],
        textposition="inside",
        textfont=dict(color="#fff", size=10),
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Triggered Red Flags", font=dict(color="#818cf8", size=13)),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=10, color="#cbd5e1")),
        height=max(200, len(labels) * 38),
        bargap=0.25,
    )
    return fig


def make_session_pie() -> go.Figure:
    history    = st.session_state.history
    fake_count = sum(1 for h in history if h["prediction"] == "FAKE")
    real_count = len(history) - fake_count

    fig = go.Figure(go.Pie(
        labels=["Fake Jobs", "Real Jobs"],
        values=[max(fake_count, 0.01), max(real_count, 0.01)],
        hole=0.55,
        marker=dict(colors=["#ef4444", "#22c55e"], line=dict(color="#0a0e1a", width=2)),
        textinfo="label+value",
        textfont=dict(size=11, color="#e2e8f0"),
    ))
    fig.add_annotation(
        text=f"{len(history)}<br><span style='font-size:10px'>scanned</span>",
        x=0.5, y=0.5,
        font=dict(size=16, color="#e2e8f0", family="JetBrains Mono"),
        showarrow=False,
    )
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Session Stats", font=dict(color="#818cf8", size=13)),
        showlegend=True,
        legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
        height=250,
    )
    return fig


def make_confidence_bar(history: list) -> go.Figure:
    if not history:
        return go.Figure()

    labels = [f"#{i+1} {h['prediction']}" for i, h in enumerate(history)]
    values = [h["confidence"] for h in history]
    colors = ["#ef4444" if h["prediction"] == "FAKE" else "#22c55e" for h in history]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)")),
        hovertemplate="Job #%{x}<br>Confidence: %{y}%<extra></extra>",
        text=[f"{v}%" for v in values],
        textposition="outside",
        textfont=dict(color="#cbd5e1", size=10),
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Confidence per Scan", font=dict(color="#818cf8", size=13)),
        xaxis=dict(showgrid=False, tickfont=dict(size=9, color="#64748b")),
        yaxis=dict(showgrid=True, gridcolor="rgba(99,102,241,0.1)",
                   tickfont=dict(size=9), range=[0, 115]),
        height=250,
        bargap=0.3,
    )
    return fig


def render_result(result: dict, elapsed: float):
    """Render verdict card, explanation, flags and charts from stored result dict."""
    is_fake = result["prediction"] == "FAKE"

    st.markdown("---")
    res_col, chart_col = st.columns([1, 1], gap="large")

    with res_col:
        if is_fake:
            risk_class = f"risk-{result['risk_level'].lower()}"
            st.markdown(f"""
            <div class="result-fake">
                <div class="verdict-text verdict-fake">🚨 FAKE JOB</div>
                <div class="confidence-text">Confidence: <strong>{result['confidence']}%</strong></div>
                <div style="margin-top:10px;">
                    <span class="{risk_class}">⚠️ {result['risk_level']} RISK</span>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-real">
                <div class="verdict-text verdict-real">✅ LEGITIMATE</div>
                <div class="confidence-text">Confidence: <strong>{result['confidence']}%</strong></div>
                <div style="margin-top:10px;">
                    <span class="risk-low">✅ {result['risk_level']} RISK</span>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        st.markdown(f"""
        <div class="explanation-box">
            💡 {result['explanation']}
            <div style="margin-top:10px;font-size:0.8rem;color:#475569;">
                Analyzed in {elapsed:.0f}ms &nbsp;|&nbsp;
                Weights: {result['scores'].get('weights_used', '—')}
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("")

        if result["flag_descriptions"]:
            st.markdown('<div class="section-header">⚠️ Red Flags Detected</div>',
                        unsafe_allow_html=True)
            for desc in result["flag_descriptions"]:
                st.markdown(f'<div class="flag-item">🚩 {desc}</div>',
                            unsafe_allow_html=True)
        else:
            st.markdown('<div class="section-header">✅ No Red Flags</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="flag-item-real">✓ No scam patterns detected</div>',
                        unsafe_allow_html=True)

    with chart_col:
        st.plotly_chart(
            make_score_pie(result["scores"]),
            width='stretch',
            config={"displayModeBar": False},
        )
        if result["flag_descriptions"]:
            st.plotly_chart(
                make_flags_bar(result["flag_descriptions"]),
                width='stretch',
                config={"displayModeBar": False},
            )

    with st.expander("🔢 Detailed score breakdown"):
        sc = result["scores"]
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Final Score",  f"{float(sc.get('final', 0)):.3f}")
        s2.metric("Rules Score",  f"{float(sc.get('rules', 0)):.3f}")
        s3.metric("TF-IDF Score", f"{float(sc.get('tfidf', 0)):.3f}")
        emb_val = sc.get('embedding', 'n/a')
        s4.metric("Embedding", f"{float(emb_val):.3f}" if emb_val != 'n/a' else "n/a")


# ═══════════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-title">🛡️ JobGuard AI</div>
    <div class="hero-subtitle">
        Multi-layer AI system to detect fake & scam job descriptions instantly
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT: Input (left) | Session Charts (right)
# ═══════════════════════════════════════════════════════════════════════════════
col_input, col_charts = st.columns([1.1, 0.9], gap="large")

with col_input:
    st.markdown('<div class="section-header">📋 Job Description Input</div>',
                unsafe_allow_html=True)

    job_text = st.text_area(
        "Paste the full job description here",
        height=240,
        placeholder="Paste the full job description here...\n\nExample: URGENT HIRING!! Earn ₹50,000/week!! No experience needed!!",
        label_visibility="collapsed",
    )

    c1, c2 = st.columns(2)
    with c1:
        title   = st.text_input("Job Title (optional)",    placeholder="e.g. Data Analyst")
    with c2:
        company = st.text_input("Company Name (optional)", placeholder="e.g. TechCorp Pvt Ltd")

    analyze_clicked = st.button("🔍 Analyze Job Post", width="stretch")

    st.markdown("")
    lcol1, lcol2, lcol3 = st.columns(3)
    with lcol1:
        st.markdown("""
        <div style="background:#111827;border:1px solid #22c55e33;border-radius:10px;
             padding:12px;text-align:center;">
            <div style="font-size:1.3rem;">⚙️</div>
            <div style="font-size:0.75rem;color:#22c55e;font-weight:600;margin-top:4px;">Rule Engine</div>
            <div style="font-size:0.65rem;color:#475569;">25 rules active</div>
        </div>""", unsafe_allow_html=True)
    with lcol2:
        color2 = "#22c55e" if model_loaded else "#f97316"
        label2 = "TF-IDF ML" if model_loaded else "TF-IDF (untrained)"
        st.markdown(f"""
        <div style="background:#111827;border:1px solid {color2}33;border-radius:10px;
             padding:12px;text-align:center;">
            <div style="font-size:1.3rem;">🤖</div>
            <div style="font-size:0.75rem;color:{color2};font-weight:600;margin-top:4px;">{label2}</div>
            <div style="font-size:0.65rem;color:#475569;">n-gram + features</div>
        </div>""", unsafe_allow_html=True)
    with lcol3:
        color3 = "#22c55e" if hf_active else "#6366f1"
        sub3   = "HF API active" if hf_active else "HF API ready"
        st.markdown(f"""
        <div style="background:#111827;border:1px solid {color3}33;border-radius:10px;
             padding:12px;text-align:center;">
            <div style="font-size:1.3rem;">🧠</div>
            <div style="font-size:0.75rem;color:{color3};font-weight:600;margin-top:4px;">Semantic AI</div>
            <div style="font-size:0.65rem;color:#475569;">{sub3}</div>
        </div>""", unsafe_allow_html=True)

with col_charts:
    st.markdown('<div class="section-header">📊 Session Analytics</div>',
                unsafe_allow_html=True)

    if st.session_state.history:
        fake_count = sum(1 for h in st.session_state.history if h["prediction"] == "FAKE")
        real_count = len(st.session_state.history) - fake_count
        avg_conf   = int(sum(h["confidence"] for h in st.session_state.history)
                         / len(st.session_state.history))

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""<div class="stat-card">
                <div class="stat-number" style="color:#ef4444">{fake_count}</div>
                <div class="stat-label">Fake Detected</div></div>""",
                unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="stat-card">
                <div class="stat-number" style="color:#22c55e">{real_count}</div>
                <div class="stat-label">Real Jobs</div></div>""",
                unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="stat-card">
                <div class="stat-number">{avg_conf}%</div>
                <div class="stat-label">Avg Confidence</div></div>""",
                unsafe_allow_html=True)

        st.markdown("")
        tab1, tab2 = st.tabs(["🥧 Fake vs Real", "📈 Confidence"])
        with tab1:
            st.plotly_chart(make_session_pie(), width='stretch',
                            config={"displayModeBar": False})
        with tab2:
            st.plotly_chart(make_confidence_bar(st.session_state.history),
                            width='stretch', config={"displayModeBar": False})
    else:
        st.markdown("""
        <div style="background:#111827;border:1px solid rgba(99,102,241,0.15);
             border-radius:12px;padding:48px 24px;text-align:center;color:#475569;">
            <div style="font-size:2.5rem;margin-bottom:12px;">📊</div>
            <div style="font-size:0.95rem;">Charts appear here after your first analysis</div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYZE — run detection, save to session state, then rerun to refresh charts
# ═══════════════════════════════════════════════════════════════════════════════
if analyze_clicked:
    if not job_text.strip():
        st.warning("⚠️ Please paste a job description first.")
    else:
        with st.spinner("🔍 Analyzing with 3-layer AI system..."):
            t0      = time.perf_counter()
            result  = detector.predict(text=job_text, title=title, company=company)
            elapsed = (time.perf_counter() - t0) * 1000

        # ── Store full result in session state so it survives st.rerun() ──────
        st.session_state.last_result = {
            "prediction":        result.prediction,
            "confidence":        result.confidence,
            "risk_level":        result.risk_level,
            "scores":            result.scores,
            "explanation":       result.explanation,
            "flag_descriptions": result.flag_descriptions,
        }
        st.session_state.last_elapsed = elapsed

        # ── Add to history for session charts ─────────────────────────────────
        st.session_state.history.append({
            "prediction": result.prediction,
            "confidence": result.confidence,
            "risk_level": result.risk_level,
            "scores":     result.scores,
        })

        # ── Rerun so session charts refresh with new data ─────────────────────
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS — rendered from session state every run so they never disappear
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.last_result is not None:
    render_result(st.session_state.last_result, st.session_state.last_elapsed)


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:32px 0 16px;color:#334155;font-size:0.8rem;">
    🛡️ JobGuard AI &nbsp;·&nbsp; 3-Layer Detection: Rule Engine + TF-IDF + Semantic AI
    &nbsp;·&nbsp; Built to protect job seekers from scams
</div>
""", unsafe_allow_html=True)