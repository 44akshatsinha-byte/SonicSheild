"""SonicShield Agent Assist Dashboard — Acoustic Liveness Intelligence.

Features real-time rule-based baseline scoring, genuine trained ML classifier scoring,
and side-by-side comparison mode across rolling 6s windows.
"""
from __future__ import annotations

import io
import os
import soundfile as sf
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.codec_crusher import get_supported_codecs
from src.pipeline import run_pipeline, PipelineError
from src.ml_classifier import get_ml_classifier
from src.pretrained_aasist import get_aasist_classifier

# Page Configuration
st.set_page_config(
    page_title="SonicShield | Acoustic Liveness Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Cybersecurity Styling
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: #070B14;
        color: #E2E8F0;
    }

    .stApp {
        background: radial-gradient(circle at 15% 20%, #0d1933 0%, #070b14 75%);
    }

    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        color: #FFFFFF;
        letter-spacing: -0.02em;
    }

    .shield-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        background: rgba(0, 242, 254, 0.12);
        border: 1px solid rgba(0, 242, 254, 0.35);
        color: #00F2FE;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .model-status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 8px;
    }
    .model-status-active {
        background: rgba(0, 230, 118, 0.15);
        border: 1px solid rgba(0, 230, 118, 0.4);
        color: #00E676;
    }
    .model-status-unloaded {
        background: rgba(148, 163, 184, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.3);
        color: #94A3B8;
    }

    .brand-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 0%, #B9D1F4 50%, #00F2FE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.1;
    }

    .brand-caption {
        color: #8A99AD;
        font-size: 0.95rem;
        margin-top: 6px;
        line-height: 1.4;
    }

    .metric-card {
        background: rgba(16, 26, 48, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        backdrop-filter: blur(12px);
        margin-bottom: 16px;
        transition: border 0.2s ease;
    }

    .metric-card:hover {
        border-color: rgba(0, 242, 254, 0.3);
    }

    .card-label {
        font-size: 0.8rem;
        color: #8A99AD;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .agent-tip-box {
        background: linear-gradient(135deg, rgba(255, 171, 0, 0.12) 0%, rgba(255, 82, 82, 0.10) 100%);
        border: 1px solid rgba(255, 171, 0, 0.35);
        border-radius: 12px;
        padding: 16px 20px;
        margin-top: 12px;
    }

    .tip-title {
        color: #FFD600;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .tip-text {
        font-size: 1.05rem;
        color: #FFF9E6;
        font-weight: 500;
        line-height: 1.4;
        font-style: italic;
    }

    .meter-container {
        margin-top: 10px;
        margin-bottom: 16px;
    }
    .meter-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.88rem;
        color: #CBD5E1;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .meter-bar-bg {
        width: 100%;
        height: 10px;
        background: #1B273D;
        border-radius: 999px;
        overflow: hidden;
    }
    .meter-bar-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.5s ease;
    }

    .limits-footer {
        background: rgba(13, 21, 38, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 14px 20px;
        font-size: 0.82rem;
        color: #7A889B;
        line-height: 1.5;
        margin-top: 36px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Helper function to convert audio array to WAV bytes for playback
def audio_to_wav_bytes(audio: np.ndarray, sr: int = 8000) -> bytes:
    buf = io.BytesIO()
    clipped = np.clip(audio, -1.0, 1.0).astype(np.float32)
    sf.write(buf, clipped, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()

# Gauge Chart Builder
def render_gauge(score: int, band: str, title: str = "") -> go.Figure:
    if band == "green":
        bar_color = "#00E676"
    elif band == "amber":
        bar_color = "#FFD600"
    else:
        bar_color = "#FF5252"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": title, "font": {"size": 14, "color": "#94A3B8"}} if title else None,
            domain={"x": [0, 1], "y": [0, 1]},
            number={"suffix": "", "font": {"size": 44, "color": "#FFFFFF", "family": "Outfit"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569", "ticks": "outside"},
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "rgba(255, 255, 255, 0.04)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(255, 82, 82, 0.15)"},
                    {"range": [40, 70], "color": "rgba(255, 214, 0, 0.15)"},
                    {"range": [70, 100], "color": "rgba(0, 230, 118, 0.15)"},
                ],
                "threshold": {
                    "line": {"color": bar_color, "width": 4},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=210,
        margin={"l": 20, "r": 20, "t": 25, "b": 10},
    )
    return fig

# Score Timeline Chart Builder
def render_timeline(windows: list[dict], show_ml: bool = False, show_aasist: bool = False) -> go.Figure:
    t_vals = [w["t"] for w in windows]
    baseline_vals = [w["baseline_score"] for w in windows]
    physics_vals = [w["physics"] * 100.0 for w in windows]
    biology_vals = [w["biology"] * 100.0 for w in windows]
    ml_vals = [w.get("ml_score") for w in windows]
    aasist_vals = [w.get("aasist_score") for w in windows]

    fig = go.Figure()

    # Band background regions
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(0, 230, 118, 0.07)", line_width=0, layer="below")
    fig.add_hrect(y0=40, y1=70, fillcolor="rgba(255, 214, 0, 0.07)", line_width=0, layer="below")
    fig.add_hrect(y0=0, y1=40, fillcolor="rgba(255, 82, 82, 0.07)", line_width=0, layer="below")

    # Threshold guidelines
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(0, 230, 118, 0.3)", line_width=1)
    fig.add_hline(y=40, line_dash="dash", line_color="rgba(255, 214, 0, 0.3)", line_width=1)

    # Subscore lines
    fig.add_trace(
        go.Scatter(
            x=t_vals,
            y=physics_vals,
            mode="lines",
            name="Room & Echo (Physics)",
            line={"color": "#38BDF8", "width": 1.8, "dash": "dot"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_vals,
            y=biology_vals,
            mode="lines",
            name="Breath & Voice (Biology)",
            line={"color": "#C084FC", "width": 1.8, "dash": "dot"},
        )
    )

    # Baseline Score Line
    fig.add_trace(
        go.Scatter(
            x=t_vals,
            y=baseline_vals,
            mode="lines+markers",
            name="Baseline Score (Rule-Based)",
            line={"color": "#00F2FE", "width": 3.0},
            marker={"size": 6, "color": "#00F2FE", "symbol": "circle"},
        )
    )

    # Pretrained AASIST Score Line (if available)
    if show_aasist and any(v is not None for v in aasist_vals):
        fig.add_trace(
            go.Scatter(
                x=t_vals,
                y=[v if v is not None else np.nan for v in aasist_vals],
                mode="lines+markers",
                name="Pretrained AASIST Score",
                line={"color": "#EC4899", "width": 3.0, "dash": "dashdot"},
                marker={"size": 6, "color": "#EC4899", "symbol": "star"},
            )
        )

    # ML Score Line (if available)
    if show_ml and any(v is not None for v in ml_vals):
        fig.add_trace(
            go.Scatter(
                x=t_vals,
                y=[v if v is not None else np.nan for v in ml_vals],
                mode="lines+markers",
                name="ML Model Score",
                line={"color": "#F59E0B", "width": 3.0, "dash": "dash"},
                marker={"size": 6, "color": "#F59E0B", "symbol": "diamond"},
            )
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(16, 26, 48, 0.4)",
        xaxis={
            "title": "Timeline (seconds)",
            "gridcolor": "rgba(255, 255, 255, 0.06)",
            "zeroline": False,
            "color": "#94A3B8",
        },
        yaxis={
            "title": "Score (0 - 100)",
            "range": [0, 105],
            "gridcolor": "rgba(255, 255, 255, 0.06)",
            "zeroline": False,
            "color": "#94A3B8",
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"color": "#CBD5E1", "size": 11},
        },
        height=280,
        margin={"l": 30, "r": 20, "t": 35, "b": 35},
    )
    return fig

def format_aasist_class(pred_class: str | None) -> tuple[str, str]:
    """Maps AASIST predicted class to user-facing labels and theme colors."""
    if pred_class == "bona_fide":
        return "Bona fide-like audio", "#00E676"
    elif pred_class == "spoof":
        return "Spoof-like audio", "#FF5252"
    else:
        return "Uncertain or insufficient evidence", "#FFD600"

# Check Model Status
ml_classifier = get_ml_classifier()
aasist_classifier = get_aasist_classifier()
supported_codecs = get_supported_codecs()

# Top Header
st.markdown('<div class="shield-badge">Acoustic Liveness Intelligence</div>', unsafe_allow_html=True)

badges = []
if aasist_classifier.is_loaded:
    badges.append('<span class="model-status-badge model-status-active">● AASIST Pretrained Active</span>')
else:
    badges.append('<span class="model-status-badge model-status-unloaded">○ AASIST Unloaded</span>')

if ml_classifier.is_loaded:
    model_name = ml_classifier.metadata.get("model_type", "Logistic Regression")
    badges.append(f'<span class="model-status-badge model-status-active">● ML Active: {model_name}</span>')
else:
    badges.append('<span class="model-status-badge model-status-unloaded">○ Custom ML Unloaded</span>')

badge_html = " ".join(badges)

st.markdown(f'<h1 class="brand-title">SONICSHIELD {badge_html}</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="brand-caption">'
    'Zero-friction acoustic deepfake & liveness detection prototype. Inspects micro-room reverberation tails, '
    'pause room tone, involuntary biological vocal dynamics, and official pretrained AASIST neural anti-spoofing '
    'under telephone-codec conditions.'
    '</p>',
    unsafe_allow_html=True,
)
st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 18px 0;'>", unsafe_allow_html=True)

# Sidebar: Controls & Codec Selection
with st.sidebar:
    st.markdown("### ⚙️ Scoring & Detection")

    detection_mode = st.radio(
        "Detection Engine",
        options=["baseline", "pretrained", "ml", "compare"],
        format_func=lambda x: {
            "baseline": "Rule-Based Baseline (Physics + Biology)",
            "pretrained": "Pretrained AASIST Model (Official SOTA)",
            "ml": "Custom ML Classifier (Acoustic Features)",
            "compare": "Multi-Engine Comparison (Side-by-Side)",
        }[x],
        index=0,
        help="Choose whether to run the transparent rule-based heuristic baseline, the pretrained AASIST deep neural network, the custom ML classifier, or all engines side-by-side.",
    )

    if detection_mode in ("pretrained", "compare") and not aasist_classifier.is_loaded:
        st.warning("⚠️ *Pretrained AASIST model not loaded. Verify models/pretrained/AASIST.pth.*")

    if detection_mode in ("ml", "compare") and not ml_classifier.is_loaded:
        st.info("ℹ️ *Custom ML model not yet loaded. Baseline will serve as primary score. Train with `python scripts/train_ml_model.py`.*")

    st.markdown("---")
    st.markdown("### 🎙️ Audio & Codec")

    # Mode selection (Ablation support)
    analysis_mode = st.selectbox(
        "Engine Ablation",
        options=["both", "physics", "biology"],
        format_func=lambda x: {
            "both": "Combined (Physics + Biology)",
            "physics": "Physics Only (Room & Echo)",
            "biology": "Biology Only (Breath & Voice)",
        }[x],
        index=0,
    )

    # Codec selection
    codec_options = ["clean", "g711"]
    if supported_codecs["amr"]:
        codec_options.append("amr")

    selected_codec = st.selectbox(
        "Telephone Codec Simulation",
        options=codec_options,
        format_func=lambda x: {
            "clean": "Clean (8 kHz Reference)",
            "g711": "G.711 μ-law (ITU-T Standard Landline)",
            "amr": "AMR-NB 12.2k (GSM Cellular)",
        }.get(x, x),
        index=1 if "g711" in codec_options else 0,
        help="Simulates authentic telephone compression before acoustic analysis.",
    )

    if not supported_codecs["amr"]:
        st.caption("ℹ️ *AMR-NB requires FFmpeg with libopencore-amrnb. G.711 μ-law native is fully active.*")

    # Network impairment toggles
    with st.expander("Line Impairments (Optional)"):
        packet_loss = st.slider("VoIP Packet Loss Rate", 0.0, 0.15, 0.0, 0.01, format="%.2f")
        add_noise = st.checkbox("Add Telephone Line Hiss", value=False)
        noise_snr = 25.0 if add_noise else None

    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 16px 0;'>", unsafe_allow_html=True)
    
    # Preloaded Smoke Audio Selector for Instant Live Demo
    st.markdown("### 🎙️ Quick Demo Audio")
    smoke_dir = "data/smoke"
    smoke_options = {"None (Upload file)": None}
    if os.path.exists(smoke_dir):
        smoke_files = [f for f in os.listdir(smoke_dir) if f.endswith(".wav")]
        for sf_name in smoke_files:
            label = sf_name.replace(".wav", "").replace("smoke_", "").replace("_", " ").title()
            smoke_options[f"Demo: {label}"] = os.path.join(smoke_dir, sf_name)

    selected_smoke = st.selectbox("Select Benchmark Sample", list(smoke_options.keys()))

# Main Area Tabs: Single Call Analysis vs Dual Recording Comparison
tab_live, tab_compare = st.tabs(["📞 Single Call Analysis", "⚖️ Dual Audio Comparison"])

# --- TAB 1: SINGLE CALL ANALYSIS ---
with tab_live:
    col_input, col_status = st.columns([2, 1])

    with col_input:
        uploaded_file = st.file_uploader(
            "Upload Caller Audio (WAV format, mono/stereo)",
            type=["wav"],
            key="single_upload",
            help="Accepts standard WAV files. Audio will be converted to mono and resampled to 8000 Hz.",
        )

    # Determine input source
    audio_source = None
    input_label = None

    if selected_smoke != "None (Upload file)" and smoke_options[selected_smoke]:
        audio_source = smoke_options[selected_smoke]
        input_label = f"Demo Sample: {selected_smoke}"
    elif uploaded_file is not None:
        audio_source = uploaded_file.getvalue()
        input_label = uploaded_file.name

    with col_status:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        analyze_clicked = st.button("🚀 Analyze Caller Audio", type="primary", use_container_width=True)

    if audio_source is not None:
        if analyze_clicked or "last_result" not in st.session_state:
            with st.spinner("Processing audio through telephone codec & acoustic engines..."):
                try:
                    result = run_pipeline(
                        file_source=audio_source,
                        codec=selected_codec,
                        mode=analysis_mode,
                        detection_mode=detection_mode,
                        packet_loss_rate=packet_loss,
                        line_noise_snr=noise_snr,
                    )
                    st.session_state["last_result"] = result
                    st.session_state["last_label"] = input_label
                except PipelineError as e:
                    st.error(f"Analysis Pipeline Error: {e}")
                except Exception as e:
                    st.error(f"Unexpected Error: {e}")

    if "last_result" in st.session_state:
        res = st.session_state["last_result"]
        lbl = st.session_state.get("last_label", "Audio")
        score = res["overall_liveness"]
        band = res["overall_band"]
        band_label = res["overall_band_label"]
        active_mode = res.get("detection_mode", "baseline")

        # Display any warnings
        if res.get("warnings"):
            for w in res["warnings"]:
                st.warning(f"⚠️ {w}")

        st.markdown(
            f"#### 🔍 Live Inspection: `{lbl}` &nbsp;|&nbsp; "
            f"Codec: `{res['codec'].upper()}` &nbsp;|&nbsp; "
            f"Mode: `{active_mode.upper()}`"
        )

        # Top Metric Row
        if active_mode == "compare":
            # Multi-Engine Comparison Mode: 3 Model Cards
            st.markdown("### ⚖️ Multi-Engine Side-by-Side Assessment")
            col_m1, col_m2, col_m3 = st.columns(3)

            with col_m1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="card-label">1. Rule-Based Baseline</div>', unsafe_allow_html=True)
                base_score = res["overall_baseline_liveness"]
                st.plotly_chart(render_gauge(base_score, res["overall_band"]), use_container_width=True, config={"displayModeBar": False})
                st.markdown(f"<div style='text-align:center; font-weight:700;'>{base_score} / 100</div>", unsafe_allow_html=True)
                st.caption("<div style='text-align:center;'>Physics (Room) + Biology (Vocal) Heuristic</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_m2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="card-label">2. Pretrained AASIST (SOTA)</div>', unsafe_allow_html=True)
                aasist_score = res.get("overall_aasist_liveness")
                if aasist_score is not None:
                    aas_band = "green" if aasist_score >= 70 else ("amber" if aasist_score >= 40 else "red")
                    st.plotly_chart(render_gauge(aasist_score, aas_band), use_container_width=True, config={"displayModeBar": False})
                    cls_lbl, cls_col = format_aasist_class(res.get("overall_aasist_class"))
                    st.markdown(f"<div style='text-align:center; font-weight:700; color:{cls_col};'>{cls_lbl}</div>", unsafe_allow_html=True)
                    st.caption(f"<div style='text-align:center;'>Bona Fide Evidence: {res.get('overall_aasist_bona_fide_score')}</div>", unsafe_allow_html=True)
                else:
                    st.info("AASIST model unloaded. Checkpoint missing.")
                st.markdown('</div>', unsafe_allow_html=True)

            with col_m3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="card-label">3. Custom ML Classifier</div>', unsafe_allow_html=True)
                if res.get("overall_ml_liveness") is not None:
                    ml_s = res["overall_ml_liveness"]
                    ml_b = "green" if ml_s >= 70 else ("amber" if ml_s >= 40 else "red")
                    st.plotly_chart(render_gauge(ml_s, ml_b), use_container_width=True, config={"displayModeBar": False})
                    st.markdown(f"<div style='text-align:center; font-weight:700;'>{ml_s} / 100</div>", unsafe_allow_html=True)
                    st.caption("<div style='text-align:center;'>18-D Handcrafted Feature Classifier</div>", unsafe_allow_html=True)
                else:
                    st.info("Custom model unloaded. Train with `python scripts/train_ml_model.py`.")
                st.markdown('</div>', unsafe_allow_html=True)

            col_meters, col_reasons = st.columns([1.2, 1.8])

        else:
            col_gauge, col_meters, col_reasons = st.columns([1.1, 1.2, 1.7])

            with col_gauge:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if active_mode == "pretrained":
                    label_txt = "Pretrained AASIST Model Score"
                elif active_mode == "ml":
                    label_txt = "ML-Derived Liveness Score"
                else:
                    label_txt = "Acoustic Liveness Score (Baseline)"

                st.markdown(f'<div class="card-label">{label_txt}</div>', unsafe_allow_html=True)
                st.plotly_chart(render_gauge(score, band), use_container_width=True, config={"displayModeBar": False})
                
                # Status Badge
                badge_color = "#00E676" if band == "green" else ("#FFD600" if band == "amber" else "#FF5252")
                st.markdown(
                    f"<div style='text-align:center; font-weight:700; color:{badge_color}; font-size:1.05rem; margin-top:-15px;'>"
                    f"{band_label}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)

        with col_meters:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">Independent Sub-Engine Readings</div>', unsafe_allow_html=True)

            phys_val = int(round(res["physics_mean"] * 100))
            bio_val = int(round(res["biology_mean"] * 100))

            # Room & Echo Meter
            st.markdown(
                f"""
                <div class="meter-container">
                    <div class="meter-header">
                        <span>🏛️ Room & Echo (Physics)</span>
                        <span style="color:#38BDF8;">{phys_val} / 100</span>
                    </div>
                    <div class="meter-bar-bg">
                        <div class="meter-bar-fill" style="width: {phys_val}%; background: linear-gradient(90deg, #0284C7, #38BDF8);"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Breath & Voice Meter
            st.markdown(
                f"""
                <div class="meter-container">
                    <div class="meter-header">
                        <span>🫁 Breath & Voice (Biology)</span>
                        <span style="color:#C084FC;">{bio_val} / 100</span>
                    </div>
                    <div class="meter-bar-bg">
                        <div class="meter-bar-fill" style="width: {bio_val}%; background: linear-gradient(90deg, #9333EA, #C084FC);"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Quick specs
            st.caption(
                f"⏱️ Duration: **{res['duration_sec']}s** &nbsp;|&nbsp; "
                f"🪟 Windows: **{res['num_windows']}** &nbsp;|&nbsp; "
                f"⚡ Latency: **{res['processing_time_ms']} ms**"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col_reasons:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">Primary Acoustic Observations</div>', unsafe_allow_html=True)
            for r in res.get("reasons", []):
                st.markdown(f"• **{r}**")

            # Agent Follow-Up Guidance on Amber/Red
            if res.get("agent_tip"):
                st.markdown(
                    f"""
                    <div class="agent-tip-box">
                        <div class="tip-title">🛡️ Agent Assist Verification Prompt</div>
                        <div class="tip-text">"{res['agent_tip']}"</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.success("✅ Acoustic metrics fall within expected natural human distribution.")
            st.markdown('</div>', unsafe_allow_html=True)

        # Dedicated Pretrained AI Anti-Spoofing Engine (AASIST) Card
        aasist_data = res.get("aasist", {})
        aasist_loaded = res.get("aasist_model_loaded", False)
        aasist_class = res.get("overall_aasist_class")
        aasist_bf_score = res.get("overall_aasist_bona_fide_score")
        aasist_live = res.get("overall_aasist_liveness")

        class_label, class_color = format_aasist_class(aasist_class if aasist_loaded else None)

        st.markdown(
            f"""
            <div class="metric-card" style="border: 1px solid rgba(236, 72, 153, 0.4); background: linear-gradient(135deg, rgba(236, 72, 153, 0.08) 0%, rgba(13, 25, 51, 0.65) 100%);">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom: 14px;">
                    <div>
                        <span class="shield-badge" style="background:rgba(236, 72, 153, 0.15); border-color:rgba(236, 72, 153, 0.4); color:#F472B6;">
                            PRETRAINED AI ANTI-SPOOFING ENGINE — AASIST
                        </span>
                        <div style="font-size: 1.22rem; font-weight: 700; color: #FFFFFF; margin-top: 4px;">
                            Official Pretrained AASIST Neural Network
                        </div>
                        <div style="font-size: 0.85rem; color: #94A3B8;">
                            Spectro-Temporal Graph Attention Network • Clova AI / ICASSP 2022 • Checkpoint: AASIST.pth
                        </div>
                    </div>
                    <div>
                        <span style="display:inline-block; padding: 6px 16px; border-radius: 999px; background: rgba(0,0,0,0.4); border: 1px solid {class_color}; color: {class_color}; font-weight: 700; font-size: 0.95rem;">
                            ● {class_label}
                        </span>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        col_a1, col_a2, col_a3, col_a4 = st.columns(4)

        with col_a1:
            st.markdown('<div class="card-label">Predicted Assessment</div>', unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:1.15rem; font-weight:700; color:{class_color};'>{class_label}</div>", unsafe_allow_html=True)
            st.caption(f"AASIST Class: `{'Class 1 (Bona Fide)' if aasist_class == 'bona_fide' else ('Class 0 (Spoof)' if aasist_class == 'spoof' else 'Unavailable')}`")

        with col_a2:
            st.markdown('<div class="card-label">Bona Fide Evidence Score</div>', unsafe_allow_html=True)
            if aasist_bf_score is not None:
                st.markdown(f"<div style='font-size:1.35rem; font-weight:800; color:#FFFFFF;'>{aasist_bf_score:.4f} <span style='font-size:0.85rem; color:#94A3B8;'>/ 1.0</span></div>", unsafe_allow_html=True)
                st.caption(f"Calibrated Liveness Index: **{aasist_live} / 100**")
            else:
                st.markdown("<div style='color:#94A3B8;'>Unavailable</div>", unsafe_allow_html=True)

        with col_a3:
            st.markdown('<div class="card-label">Raw Output Logits</div>', unsafe_allow_html=True)
            raw_logits = aasist_data.get("raw_score")
            if raw_logits and len(raw_logits) == 2:
                st.markdown(f"<code style='font-size:0.82rem;'>Spoof: {raw_logits[0]:+.3f} | Real: {raw_logits[1]:+.3f}</code>", unsafe_allow_html=True)
                st.caption("Derived via Softmax: $P(\\text{bona fide}) = \\frac{e^{z_1}}{e^{z_0}+e^{z_1}}$")
            else:
                st.markdown("<div style='color:#94A3B8;'>None</div>", unsafe_allow_html=True)

        with col_a4:
            st.markdown('<div class="card-label">Codec & Audio Condition</div>', unsafe_allow_html=True)
            st.markdown(f"**{res['codec'].upper()}** ➔ **16 kHz Mono**")
            st.caption("Input: exactly 64,600 samples (~4.04s) via deterministic tiling")

        st.markdown(
            """
            <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.06); font-size: 0.78rem; color: #8A99AD;">
                ⚠️ <em>Experimental AASIST Model Notice: Pretrained on ASVspoof 2019 Logical Access studio audio. Resampling telephone audio does not restore discarded high frequencies. Scores reflect experimental neural pattern evidence, not guaranteed biometric identity.</em>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Timeline Chart Row
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Rolling Analysis Timeline (6s Windows @ 1s Hop)</div>', unsafe_allow_html=True)
        show_ml_line = (active_mode in ("ml", "compare") or res.get("overall_ml_liveness") is not None)
        show_aasist_line = (active_mode in ("pretrained", "compare") or res.get("overall_aasist_liveness") is not None)
        st.plotly_chart(render_timeline(res["windows"], show_ml=show_ml_line, show_aasist=show_aasist_line), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

        # Audio Playback
        col_audio1, col_audio2 = st.columns(2)
        with col_audio1:
            if "raw_audio" in res:
                st.markdown("🔊 **Original Preprocessed Audio (8 kHz Mono)**")
                st.audio(audio_to_wav_bytes(res["raw_audio"]), format="audio/wav")
        with col_audio2:
            if "crushed_audio" in res:
                st.markdown(f"📻 **Telephone Codec Output ({res['codec'].upper()})**")
                st.audio(audio_to_wav_bytes(res["crushed_audio"]), format="audio/wav")

    else:
        st.info("👆 Please upload a WAV file or select a Demo Benchmark Sample from the sidebar, then click **Analyze Caller Audio**.")

# --- TAB 2: DUAL AUDIO COMPARISON ---
with tab_compare:
    st.markdown("### ⚖️ Side-by-Side Audio Comparison")
    st.caption("Compare two audio recordings (e.g. consented human speaker vs. voice clone) through the same telephone codec.")

    col_cmp_a, col_cmp_b = st.columns(2)

    with col_cmp_a:
        st.markdown("#### Recording A (e.g., Reference Human)")
        file_a = st.file_uploader("Upload Audio A (WAV)", type=["wav"], key="cmp_file_a")
        smoke_a = st.selectbox(
            "Or select benchmark sample A",
            list(smoke_options.keys()),
            key="cmp_smoke_a",
            index=1 if len(smoke_options) > 1 else 0,
        )

    with col_cmp_b:
        st.markdown("#### Recording B (e.g., Suspected Clone)")
        file_b = st.file_uploader("Upload Audio B (WAV)", type=["wav"], key="cmp_file_b")
        smoke_b = st.selectbox(
            "Or select benchmark sample B",
            list(smoke_options.keys()),
            key="cmp_smoke_b",
            index=2 if len(smoke_options) > 2 else 0,
        )

    cmp_btn = st.button("⚖️ Run Side-by-Side Comparison", type="primary")

    src_a = file_a.getvalue() if file_a else (smoke_options[smoke_a] if smoke_a != "None (Upload file)" else None)
    src_b = file_b.getvalue() if file_b else (smoke_options[smoke_b] if smoke_b != "None (Upload file)" else None)

    if cmp_btn and src_a and src_b:
        with st.spinner("Analyzing both recordings..."):
            res_a = run_pipeline(src_a, codec=selected_codec, mode=analysis_mode, detection_mode=detection_mode)
            res_b = run_pipeline(src_b, codec=selected_codec, mode=analysis_mode, detection_mode=detection_mode)

            st.markdown("---")
            col_res_a, col_res_b = st.columns(2)

            with col_res_a:
                st.markdown(f"### Recording A: **{res_a['overall_liveness']} / 100**")
                st.plotly_chart(render_gauge(res_a["overall_liveness"], res_a["overall_band"]), use_container_width=True)
                st.markdown(f"• **Baseline Score:** `{res_a['overall_baseline_liveness']}`")
                if res_a.get("overall_aasist_bona_fide_score") is not None:
                    cls_a, _ = format_aasist_class(res_a.get("overall_aasist_class"))
                    st.markdown(f"• **Pretrained AASIST:** `{cls_a}` (score: `{res_a.get('overall_aasist_bona_fide_score')}`)")
                if res_a.get("overall_ml_liveness") is not None:
                    st.markdown(f"• **ML Model Score:** `{res_a['overall_ml_liveness']}`")
                st.markdown(f"• **Room & Echo (Physics):** `{int(res_a['physics_mean']*100)}%`")
                st.markdown(f"• **Breath & Voice (Biology):** `{int(res_a['biology_mean']*100)}%`")
                st.markdown("**Top Reasons:**")
                for r in res_a["reasons"]:
                    st.markdown(f"- {r}")

            with col_res_b:
                st.markdown(f"### Recording B: **{res_b['overall_liveness']} / 100**")
                st.plotly_chart(render_gauge(res_b["overall_liveness"], res_b["overall_band"]), use_container_width=True)
                st.markdown(f"• **Baseline Score:** `{res_b['overall_baseline_liveness']}`")
                if res_b.get("overall_aasist_bona_fide_score") is not None:
                    cls_b, _ = format_aasist_class(res_b.get("overall_aasist_class"))
                    st.markdown(f"• **Pretrained AASIST:** `{cls_b}` (score: `{res_b.get('overall_aasist_bona_fide_score')}`)")
                if res_b.get("overall_ml_liveness") is not None:
                    st.markdown(f"• **ML Model Score:** `{res_b['overall_ml_liveness']}`")
                st.markdown(f"• **Room & Echo (Physics):** `{int(res_b['physics_mean']*100)}%`")
                st.markdown(f"• **Breath & Voice (Biology):** `{int(res_b['biology_mean']*100)}%`")
                st.markdown("**Top Reasons:**")
                for r in res_b["reasons"]:
                    st.markdown(f"- {r}")

# Honest Limitations Footer
st.markdown(
    """
    <div class="limits-footer">
        <strong>⚠️ Experimental Prototype Limitations & Ethical Notice:</strong>
        <br>
        SonicShield evaluates acoustic indicators (room reverberation tails, ambient pause noise floor, and vocal perturbation) 
        using transparent rule-based heuristics and optional supervised machine-learning models. These features can naturally overlap 
        between genuine human callers and synthetic audio. Quiet acoustic environments, studio noise gates, speaker headsets, or voice compression 
        may lower scores. Conversely, replayed synthetic audio played through an external speaker may capture genuine physical room reflections. 
        A liveness score does <strong>not</strong> constitute definitive proof of identity or fraud. All scores are experimental indicators 
        designed to guide neutral conversational human verification.
    </div>
    """,
    unsafe_allow_html=True,
)
