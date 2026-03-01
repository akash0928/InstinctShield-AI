"""
VoxShield AI - Streamlit Frontend
Scam Call & Deepfake Voice Detection System
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import tempfile
import time

import streamlit as st

# ─── Page Config (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="VoxShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Path Setup ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* Base theme */
:root {
    --bg-dark: #0a0d14;
    --bg-card: #111520;
    --bg-card2: #161b28;
    --accent-cyan: #00e5ff;
    --accent-red: #ff3b3b;
    --accent-yellow: #ffd600;
    --accent-green: #00e676;
    --text-primary: #e8eaf6;
    --text-secondary: #7986a8;
    --border: rgba(0, 229, 255, 0.15);
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-dark) !important;
    color: var(--text-primary) !important;
    font-family: 'Syne', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background-image: 
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0, 229, 255, 0.06) 0%, transparent 60%),
        radial-gradient(ellipse 50% 30% at 80% 80%, rgba(255, 59, 59, 0.04) 0%, transparent 50%);
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: var(--bg-card) !important; }
[data-testid="stFileUploader"] { 
    background: var(--bg-card2) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-cyan) !important;
}

/* Metric cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
}
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 36px;
    font-weight: 800;
    line-height: 1;
}
.metric-value.cyan { color: var(--accent-cyan); }
.metric-value.red { color: var(--accent-red); }
.metric-value.yellow { color: var(--accent-yellow); }
.metric-value.green { color: var(--accent-green); }

/* Risk badge */
.risk-badge {
    display: inline-block;
    padding: 10px 28px;
    border-radius: 50px;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
}
.risk-safe { background: rgba(0, 230, 118, 0.15); color: #00e676; border: 1px solid rgba(0, 230, 118, 0.4); }
.risk-suspicious { background: rgba(255, 214, 0, 0.15); color: #ffd600; border: 1px solid rgba(255, 214, 0, 0.4); }
.risk-high { background: rgba(255, 59, 59, 0.15); color: #ff3b3b; border: 1px solid rgba(255, 59, 59, 0.4); }

/* Score bar */
.score-bar-container {
    background: rgba(255,255,255,0.05);
    border-radius: 50px;
    height: 8px;
    width: 100%;
    overflow: hidden;
    margin-top: 8px;
}
.score-bar-fill {
    height: 100%;
    border-radius: 50px;
    transition: width 0.5s ease;
}

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent-cyan);
    margin: 28px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

/* Transcript */
.transcript-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    line-height: 1.8;
    color: var(--text-primary);
    max-height: 200px;
    overflow-y: auto;
}

/* Pattern tag */
.pattern-tag {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 50px;
    font-size: 12px;
    font-family: 'Space Mono', monospace;
    margin: 4px;
    background: rgba(255, 59, 59, 0.12);
    color: #ff8a80;
    border: 1px solid rgba(255, 59, 59, 0.3);
}

/* Recommendation */
.rec-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 14px;
    background: var(--bg-card2);
    border-radius: 8px;
    margin-bottom: 6px;
    font-size: 14px;
    border-left: 2px solid var(--accent-cyan);
}

/* Upload zone */
.upload-hint {
    text-align: center;
    color: var(--text-secondary);
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Flagged sentence */
.flagged-sentence {
    background: rgba(255, 59, 59, 0.08);
    border-left: 3px solid var(--accent-red);
    padding: 8px 12px;
    border-radius: 0 8px 8px 0;
    margin: 6px 0;
    font-size: 13px;
    font-family: 'Space Mono', monospace;
}
.flag-chip {
    display: inline-block;
    background: rgba(255, 59, 59, 0.2);
    color: #ff8a80;
    font-size: 10px;
    padding: 1px 8px;
    border-radius: 50px;
    margin-right: 4px;
    font-family: 'Space Mono', monospace;
}

/* Logo / Title */
.logo-text {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 42px;
    background: linear-gradient(135deg, #00e5ff 0%, #7c4dff 50%, #00e5ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
}

/* Dividers */
hr { border-color: var(--border) !important; }

/* Streamlit overrides */
.stProgress > div > div { background: var(--accent-cyan) !important; }
.stButton > button {
    background: linear-gradient(135deg, #00e5ff22, #7c4dff22) !important;
    border: 1px solid var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
    font-family: 'Space Mono', monospace !important;
    letter-spacing: 1px !important;
    border-radius: 8px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00e5ff44, #7c4dff44) !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
}
[data-testid="stMarkdownContainer"] p { color: var(--text-primary); }
</style>
""", unsafe_allow_html=True)


# ─── Header ─────────────────────────────────────────────────────────────────
col_logo, col_mode = st.columns([3, 1])
with col_logo:
    st.markdown('<div class="logo-text">🛡️ VoxShield AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#7986a8; font-family: Space Mono, monospace; font-size:12px; letter-spacing:2px; margin-top:0;">'
        'SCAM CALL & DEEPFAKE VOICE DETECTION SYSTEM'
        '</p>',
        unsafe_allow_html=True
    )

with col_mode:
    st.markdown('<br>', unsafe_allow_html=True)
    mode = st.radio(
        "Mode",
        ["📁 Upload Mode", "🎙️ Live Mode"],
        index=0,
        horizontal=False,
        label_visibility="collapsed"
    )

st.markdown("---")


# ─── Model Warmup (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_pipeline():
    """Load all models once and cache them."""
    with st.spinner("🔧 Initializing VoxShield AI — loading models (first run only)..."):
        from pipeline import warm_up_models
        warm_up_models()
    return True


# ─── Live Mode Placeholder ───────────────────────────────────────────────────
if "Live" in mode:
    st.markdown("""
    <div style="text-align:center; padding: 80px 20px;">
        <div style="font-size:64px;">🎙️</div>
        <div style="font-family: Space Mono, monospace; color: #7986a8; font-size:13px; letter-spacing:3px; margin-top:16px;">
            LIVE MODE — COMING SOON
        </div>
        <div style="color:#4a5568; font-size:13px; margin-top:12px;">
            Real-time microphone monitoring with live risk scoring will be available in the next release.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─── Upload Mode ─────────────────────────────────────────────────────────────
col_upload, col_demo = st.columns([2, 1])

with col_upload:
    st.markdown('<div class="section-header">UPLOAD AUDIO FILE</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload audio",
        type=["mp3", "wav", "flac", "ogg"],
        label_visibility="collapsed"
    )
    st.markdown('<div class="upload-hint">Supported: MP3, WAV, FLAC, OGG · Max 60 seconds</div>', unsafe_allow_html=True)

with col_demo:
    st.markdown('<div class="section-header">DEMO SAMPLES</div>', unsafe_allow_html=True)
    demo_choice = st.selectbox(
        "Demo",
        ["-- Select demo --", "OTP Scam Call", "Police Impersonation", "Lottery Fraud", "Normal Call"],
        label_visibility="collapsed"
    )

    # Demo text samples for testing without real audio
    DEMO_TRANSCRIPTS = {
        "OTP Scam Call": (
            "Hello, this is calling from your bank's security department. We have detected suspicious "
            "transactions on your account. Your account will be blocked immediately unless you verify "
            "your identity right now. Please share the OTP sent to your registered mobile number. "
            "Do not tell anyone about this call. Stay on the line and send me the OTP now."
        ),
        "Police Impersonation": (
            "This is officer Singh from the CBI narcotics bureau. A package registered in your name "
            "was intercepted at Mumbai airport containing illegal substances. An arrest warrant has been "
            "issued. Do not contact your family or lawyer. You have 30 minutes to pay a clearance fine "
            "to avoid immediate arrest. Transfer the amount to the account we provide. Act right now."
        ),
        "Lottery Fraud": (
            "Congratulations! Your number has been selected in the national digital lottery. You have won "
            "ten lakh rupees. This is your last chance to claim before the offer expires today. To release "
            "your prize money, you need to pay a processing fee of two thousand rupees urgently. "
            "Do not share this with anyone else as slots are limited."
        ),
        "Normal Call": (
            "Hello, I am calling from City Hospital to remind you about your medical appointment scheduled "
            "for tomorrow morning at 10 AM. Please bring your previous reports and insurance card. "
            "If you need to reschedule, please call us back at the hospital number. Thank you and have a good day."
        )
    }


def _get_score_color(value: float) -> str:
    if value < 0.33:
        return "#00e676"
    elif value < 0.66:
        return "#ffd600"
    else:
        return "#ff3b3b"


def _render_score_bar(value: float, color: str):
    pct = int(value * 100)
    st.markdown(f"""
    <div class="score-bar-container">
        <div class="score-bar-fill" style="width:{pct}%; background:{color};"></div>
    </div>
    """, unsafe_allow_html=True)


def render_results(report: dict):
    """Render the full analysis results panel."""
    summary = report["summary"]
    scam = report["scam_analysis"]
    voice = report["voice_analysis"]
    manip = report["manipulation_analysis"]

    risk_level = summary["risk_level"]
    risk_score = summary["risk_score"]

    # Badge class
    badge_class = {
        "SAFE": "risk-safe",
        "SUSPICIOUS": "risk-suspicious",
        "HIGH RISK": "risk-high"
    }.get(risk_level, "risk-safe")

    # ─── Risk Overview ───────────────────────────────────────────
    st.markdown('<div class="section-header">RISK ASSESSMENT</div>', unsafe_allow_html=True)

    col_badge, col_score, col_dur = st.columns([2, 1, 1])
    with col_badge:
        st.markdown(f'<span class="risk-badge {badge_class}">{risk_level}</span>', unsafe_allow_html=True)
    with col_score:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Risk Score</div>
            <div class="metric-value {'red' if risk_score >= 70 else 'yellow' if risk_score >= 30 else 'green'}">{risk_score:.0f}</div>
            <div style="color:#7986a8; font-size:11px;">/100</div>
        </div>
        """, unsafe_allow_html=True)
    with col_dur:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Duration</div>
            <div class="metric-value cyan">{summary['audio_duration_seconds']:.0f}s</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Signal Breakdown ───────────────────────────────────────
    st.markdown('<div class="section-header">SIGNAL BREAKDOWN</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    scam_val = scam["scam_probability"]
    deep_val = voice["deepfake_probability"]
    manip_val = manip["manipulation_score"]

    with c1:
        color = _get_score_color(scam_val)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Scam Probability</div>
            <div class="metric-value" style="color:{color}">{scam_val*100:.0f}%</div>
            <div style="color:#7986a8; font-size:11px; margin-top:4px;">{scam['predicted_label']}</div>
        </div>
        """, unsafe_allow_html=True)
        _render_score_bar(scam_val, color)

    with c2:
        color = _get_score_color(deep_val)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Voice Synthetic</div>
            <div class="metric-value" style="color:{color}">{deep_val*100:.0f}%</div>
            <div style="color:#7986a8; font-size:11px; margin-top:4px;">{'Synthetic' if voice['is_synthetic'] else 'Human'}</div>
        </div>
        """, unsafe_allow_html=True)
        _render_score_bar(deep_val, color)

    with c3:
        color = _get_score_color(manip_val)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Manipulation Score</div>
            <div class="metric-value" style="color:{color}">{manip_val*100:.0f}%</div>
            <div style="color:#7986a8; font-size:11px; margin-top:4px;">{len(manip['detected_patterns'])} pattern(s)</div>
        </div>
        """, unsafe_allow_html=True)
        _render_score_bar(manip_val, color)

    # ─── Explanation ────────────────────────────────────────────
    st.markdown('<div class="section-header">ANALYSIS SUMMARY</div>', unsafe_allow_html=True)
    explanation = report.get("explanation", "")
    st.markdown(explanation)

    # ─── Manipulation Patterns ─────────────────────────────────
    if manip["detected_patterns"]:
        st.markdown('<div class="section-header">DETECTED MANIPULATION PATTERNS</div>', unsafe_allow_html=True)
        from utils.manipulation import format_flag_name
        tags_html = "".join([f'<span class="pattern-tag">{format_flag_name(p)}</span>' for p in manip["detected_patterns"]])
        st.markdown(f'<div>{tags_html}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ─── Transcript ─────────────────────────────────────────────
    st.markdown('<div class="section-header">TRANSCRIPT</div>', unsafe_allow_html=True)
    transcript_text = report.get("transcript", "")

    # Show flagged sentences with highlights
    flagged = manip.get("flagged_sentences", [])
    if flagged:
        st.markdown("**Flagged segments:**")
        for item in flagged:
            flag_chips = "".join([f'<span class="flag-chip">{f.replace("_", " ")}</span>' for f in item["flags"]])
            st.markdown(f"""
            <div class="flagged-sentence">{flag_chips}<br>{item['sentence']}</div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f'<div class="transcript-box">{transcript_text}</div>', unsafe_allow_html=True)

    # ─── Recommendations ────────────────────────────────────────
    recommendations = report.get("recommendations", [])
    if recommendations:
        st.markdown('<div class="section-header">RECOMMENDATIONS</div>', unsafe_allow_html=True)
        for rec in recommendations:
            st.markdown(f'<div class="rec-item">→ {rec}</div>', unsafe_allow_html=True)

    # ─── Timing Info ────────────────────────────────────────────
    timings = report.get("timings", {})
    total = report.get("total_time", 0)
    with st.expander("⏱️ Performance Details"):
        cols = st.columns(len(timings) + 1)
        for i, (k, v) in enumerate(timings.items()):
            cols[i].metric(k.upper(), f"{v}s")
        cols[-1].metric("TOTAL", f"{total}s")

    # ─── Raw JSON ───────────────────────────────────────────────
    with st.expander("📋 Raw JSON Report"):
        display_report = {k: v for k, v in report.items() if k not in ["highlighted_transcript"]}
        st.json(display_report)


# ─── Analysis ────────────────────────────────────────────────────────────────
analyze_col, _ = st.columns([1, 2])
run_analysis = False
demo_transcript = None

with analyze_col:
    if uploaded_file is not None or (demo_choice and demo_choice != "-- Select demo --"):
        if st.button("🔍 Analyze Audio", use_container_width=True):
            run_analysis = True


if run_analysis:
    # Warm up models
    _pipeline_ready = get_pipeline()

    if demo_choice and demo_choice != "-- Select demo --" and uploaded_file is None:
        # Demo mode: use preset transcript, synthesize fake audio info
        demo_transcript = DEMO_TRANSCRIPTS.get(demo_choice, "")
        with st.spinner("Analyzing demo audio..."):
            from utils.manipulation import analyze_manipulation
            from utils.risk_engine import build_risk_report
            from utils.explainability import generate_risk_explanation, get_recommendations, highlight_flagged_sentences
            import config

            manip_result = analyze_manipulation(demo_transcript, config.MANIPULATION_PATTERNS)
            # Simulate scores for demo
            scam_result = {
                "scam_probability": 0.85 if "Scam" in demo_choice or "Fraud" in demo_choice or "OTP" in demo_choice or "Police" in demo_choice else 0.08,
                "predicted_label": "High Risk" if "Scam" in demo_choice or "Fraud" in demo_choice or "OTP" in demo_choice or "Police" in demo_choice else "Normal",
                "class_probabilities": {}
            }
            deepfake_result = {
                "deepfake_probability": 0.55 if demo_choice != "Normal Call" else 0.12,
                "is_synthetic": demo_choice != "Normal Call",
                "method": "demo_simulation"
            }

            report = build_risk_report(
                transcript=demo_transcript,
                scam_result=scam_result,
                deepfake_result=deepfake_result,
                manipulation_result=manip_result,
                audio_duration=35.0
            )
            report["explanation"] = generate_risk_explanation(report)
            report["recommendations"] = get_recommendations(report)
            report["highlighted_transcript"] = highlight_flagged_sentences(
                demo_transcript,
                manip_result.get("flagged_sentences", [])
            )
            report["timings"] = {"preprocess": 0.1, "stt": 0.0, "scam": 0.2, "deepfake": 0.1, "manipulation": 0.05}
            report["total_time"] = 0.45

        st.success(f"✅ Demo analysis complete for: **{demo_choice}**")
        st.markdown("---")
        render_results(report)

    elif uploaded_file is not None:
        # Real audio analysis
        tmp_path = None
        try:
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(step: str, progress: float):
                progress_bar.progress(min(int(progress * 100), 100))
                status_text.markdown(
                    f'<span style="font-family: Space Mono, monospace; font-size:12px; color:#00e5ff;">'
                    f'⚙️ {step}</span>',
                    unsafe_allow_html=True
                )

            from pipeline import analyze_audio_file
            report = analyze_audio_file(tmp_path, progress_callback=progress_callback)

            progress_bar.progress(100)
            status_text.empty()

            st.success(f"✅ Analysis complete in **{report['total_time']}s** — File: **{uploaded_file.name}**")
            st.markdown("---")
            render_results(report)

        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    else:
        st.warning("Please upload an audio file or select a demo sample first.")


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#4a5568; font-family: Space Mono, monospace; font-size:10px; letter-spacing:2px; padding:12px;">
    VOXSHIELD AI · OPEN SOURCE · LOCAL INFERENCE · NO CLOUD DEPENDENCIES
</div>
""", unsafe_allow_html=True)
