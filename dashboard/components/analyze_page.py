"""SonicShield Audio Analysis Page Component.

Prioritizes:
1. Unified container card for audio upload & demo sample selector
2. Audio signal metadata grid
3. Cohesive analysis controls panel
4. Main detection result with cautious labels
5. AASIST primary neural card & supporting diagnostics
6. Audio waveform in dark technical console & audio players
7. Side-by-side comparison mode
"""
from __future__ import annotations

import os
import io
import time
import soundfile as sf
import numpy as np
import streamlit as st

from src.pipeline import run_pipeline, PipelineError
from src.codec_crusher import get_supported_codecs
from src.pretrained_aasist import get_aasist_classifier
from dashboard.components.charts import (
    render_merkur_gauge,
    render_timeline_chart,
    render_dark_waveform,
    audio_to_wav_bytes,
)

def format_cautious_outcome(score: int, aasist_class: str | None, active_mode: str) -> tuple[str, str, str, str]:
    """Generates cautious, scientifically honest classification verdicts."""
    if active_mode == "pretrained" and aasist_class:
        if aasist_class == "bona_fide":
            return (
                "Model Leans Bona-Fide",
                "Neural spectro-temporal features align more closely with genuine human vocal patterns.",
                "bona-fide",
                "#13A89E",
            )
        elif aasist_class == "spoof":
            return (
                "Model Leans Synthetic",
                "Neural anti-spoofing engine detected spectro-temporal patterns associated with synthetic or cloned audio.",
                "synthetic",
                "#D95757",
            )
        else:
            return (
                "Inconclusive / Needs Further Review",
                "The neural engine returned borderline evidence. Neutral human verification recommended.",
                "inconclusive",
                "#D99A35",
            )
    else:
        if score >= 70:
            return (
                "Model Leans Bona-Fide",
                "Acoustic reverberation and vocal perturbation indicators fall within expected natural human ranges.",
                "bona-fide",
                "#13A89E",
            )
        elif score <= 40:
            return (
                "Model Leans Synthetic",
                "Acoustic room indicators or vocal dynamics suggest an unnatural or synthetically generated signal.",
                "synthetic",
                "#D95757",
            )
        else:
            return (
                "Inconclusive / Needs Further Review",
                "Acoustic metrics fall in an intermediate zone. Secondary verification recommended.",
                "inconclusive",
                "#D99A35",
            )

def render_analyze_page():
    """Renders the balanced, cohesive 4-step guided analysis workflow and results."""
    aasist_clf = get_aasist_classifier()
    supported_codecs = get_supported_codecs()

    # Preloaded demo benchmark samples
    smoke_dir = "data/smoke"
    smoke_options = {"None (Upload file)": None}
    if os.path.exists(smoke_dir):
        for f in sorted(os.listdir(smoke_dir)):
            if f.endswith(".wav"):
                lbl = f.replace(".wav", "").replace("smoke_", "").replace("_", " ").title()
                smoke_options[f"Demo: {lbl}"] = os.path.join(smoke_dir, f)

    # Sub-tabs: Single Audio vs Dual Comparison
    analysis_tab, compare_tab = st.tabs(["Single Audio Inspection", "Dual Audio Comparison"])

    with analysis_tab:
        # Check queued source from overview
        default_audio_source = st.session_state.get("queued_source", None)
        default_label = st.session_state.get("queued_name", None)

        col_w1, col_w2 = st.columns([1.5, 1.3])

        with col_w1:
            # Step 1: Upload Audio Recording inside cohesive container
            with st.container(border=True):
                col_h1, col_h2 = st.columns([3, 1])
                with col_h1:
                    st.markdown(
                        """
                        <div style="font-size: 0.92rem; font-weight: 700; color: #14283D; margin-bottom: 2px;">
                            1. Audio Ingress & Input
                        </div>
                        <div style="font-size: 0.74rem; color: #707C8B; margin-bottom: 8px;">
                            Upload a WAV voice recording or select a preloaded benchmark audio file.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col_h2:
                    st.markdown(
                        '<div style="text-align: right;"><span class="ss-spec-pill">Step 1</span></div>',
                        unsafe_allow_html=True,
                    )

                uploaded_file = st.file_uploader(
                    "Upload WAV Recording",
                    type=["wav"],
                    key="analysis_file_uploader_v3",
                    label_visibility="collapsed",
                )

                st.markdown(
                    "<div style='font-size: 0.74rem; font-weight: 600; color: #14283D; margin: 8px 0 2px 0;'>Or Choose Preloaded Demo Sample:</div>",
                    unsafe_allow_html=True,
                )
                selected_smoke = st.selectbox(
                    "Select Benchmark Demo Sample",
                    list(smoke_options.keys()),
                    key="analysis_smoke_select_v3",
                    label_visibility="collapsed",
                )

                audio_data = None
                audio_label = None

                if uploaded_file is not None:
                    audio_data = uploaded_file.getvalue()
                    audio_label = uploaded_file.name
                    if "queued_source" in st.session_state:
                        del st.session_state["queued_source"]
                elif selected_smoke != "None (Upload file)" and smoke_options[selected_smoke]:
                    audio_data = smoke_options[selected_smoke]
                    audio_label = selected_smoke
                    if "queued_source" in st.session_state:
                        del st.session_state["queued_source"]
                elif default_audio_source is not None:
                    audio_data = default_audio_source
                    audio_label = default_label

            # Step 2: Audio Signal Metadata inside cohesive container
            with st.container(border=True):
                col_mh1, col_mh2 = st.columns([3, 1])
                with col_mh1:
                    st.markdown(
                        """
                        <div style="font-size: 0.92rem; font-weight: 700; color: #14283D; margin-bottom: 2px;">
                            2. Audio Signal Metadata
                        </div>
                        <div style="font-size: 0.74rem; color: #707C8B; margin-bottom: 8px;">
                            Header diagnostics and physical signal properties.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col_mh2:
                    st.markdown(
                        '<div style="text-align: right;"><span class="ss-spec-pill">Step 2</span></div>',
                        unsafe_allow_html=True,
                    )

                if audio_data is not None:
                    try:
                        if isinstance(audio_data, str):
                            raw_audio, sr = sf.read(audio_data)
                            fsize = os.path.getsize(audio_data)
                        else:
                            raw_audio, sr = sf.read(io.BytesIO(audio_data))
                            fsize = len(audio_data)

                        duration_sec = len(raw_audio) / float(sr)
                        num_channels = raw_audio.shape[1] if raw_audio.ndim > 1 else 1

                        meta_html = f"""
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; font-size: 0.76rem;">
                            <div style="background: #F8FAF9; padding: 6px 9px; border-radius: 6px; border: 1px solid #E3E8ED;">
                                <span style="color: #707C8B;">Audio Source:</span><br>
                                <strong style="color: #14283D;">{audio_label}</strong>
                            </div>
                            <div style="background: #F8FAF9; padding: 6px 9px; border-radius: 6px; border: 1px solid #E3E8ED;">
                                <span style="color: #707C8B;">Signal Duration:</span><br>
                                <strong style="font-family: 'JetBrains Mono';">{duration_sec:.2f} seconds</strong>
                            </div>
                            <div style="background: #F8FAF9; padding: 6px 9px; border-radius: 6px; border: 1px solid #E3E8ED;">
                                <span style="color: #707C8B;">Sample Rate:</span><br>
                                <strong style="font-family: 'JetBrains Mono';">{sr} Hz ({'Mono' if num_channels == 1 else 'Stereo'})</strong>
                            </div>
                            <div style="background: #F8FAF9; padding: 6px 9px; border-radius: 6px; border: 1px solid #E3E8ED;">
                                <span style="color: #707C8B;">Uncompressed Size:</span><br>
                                <strong style="font-family: 'JetBrains Mono';">{fsize / 1024:.1f} KB</strong>
                            </div>
                        </div>
                        """
                        st.markdown(meta_html, unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Could not parse WAV headers: {e}")
                else:
                    st.markdown(
                        "<p style='font-size:0.78rem; color:#707C8B; margin:0;'>Upload a WAV file or pick a benchmark demo above to inspect signal telemetry.</p>",
                        unsafe_allow_html=True,
                    )

        with col_w2:
            # Step 3: Analysis Controls inside cohesive container
            with st.container(border=True):
                col_ch1, col_ch2 = st.columns([3, 1])
                with col_ch1:
                    st.markdown(
                        """
                        <div style="font-size: 0.92rem; font-weight: 700; color: #14283D; margin-bottom: 2px;">
                            3. Analysis Controls
                        </div>
                        <div style="font-size: 0.74rem; color: #707C8B; margin-bottom: 8px;">
                            Select detection engine mode, telephony codec, and channel impairments.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col_ch2:
                    st.markdown(
                        '<div style="text-align: right;"><span class="ss-spec-pill">Step 3</span></div>',
                        unsafe_allow_html=True,
                    )

                detection_mode = st.selectbox(
                    "Scoring Mode",
                    options=["pretrained", "compare", "baseline", "ml"],
                    format_func=lambda x: {
                        "pretrained": "Pretrained AASIST Model (Primary Neural Engine)",
                        "compare": "Multi-Engine Comparison (All Engines Side-by-Side)",
                        "baseline": "Rule-Based Baseline (Physics + Biology)",
                        "ml": "Custom ML Classifier (18 Acoustic Features)",
                    }[x],
                    index=0,
                )

                codec_choices = ["g711", "clean"]
                if supported_codecs.get("amr"):
                    codec_choices.append("amr")

                selected_codec = st.selectbox(
                    "Telephony Codec Simulation",
                    options=codec_choices,
                    format_func=lambda x: {
                        "g711": "G.711 μ-law (Standard PSTN Landline)",
                        "clean": "Clean (8 kHz Linear Reference)",
                        "amr": "AMR-NB 12.2k (GSM Cellular)",
                    }.get(x, x),
                    index=0,
                )

                analysis_mode = st.selectbox(
                    "Ablation Focus",
                    options=["both", "physics", "biology"],
                    format_func=lambda x: {
                        "both": "Combined Diagnostics (Physics + Biology)",
                        "physics": "Physics Engine Only (Room & Echo)",
                        "biology": "Biology Engine Only (Vocal Dynamics)",
                    }[x],
                    index=0,
                )

                with st.expander("Telephony Network Impairments (Optional)"):
                    pkt_loss = st.slider("VoIP Packet Loss Rate", 0.0, 0.15, 0.0, 0.01, format="%.2f")
                    add_hiss = st.checkbox("Telephone Line Hiss", value=False)
                    noise_snr = 25.0 if add_hiss else None

                st.markdown("<hr style='border: 0; border-top: 1px solid #EEF1F4; margin: 10px 0;'>", unsafe_allow_html=True)

                run_btn = st.button("Run Audio Analysis", icon=":material/play_arrow:", type="primary", use_container_width=True)

        # Trigger Processing
        if run_btn and audio_data is not None:
            with st.spinner("Executing telephony compression and multi-engine inspection..."):
                try:
                    result = run_pipeline(
                        file_source=audio_data,
                        codec=selected_codec,
                        mode=analysis_mode,
                        detection_mode=detection_mode,
                        packet_loss_rate=pkt_loss,
                        line_noise_snr=noise_snr,
                    )
                    st.session_state["active_result"] = result
                    st.session_state["active_label"] = audio_label
                    st.session_state["active_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

                    if "analysis_history" not in st.session_state:
                        st.session_state["analysis_history"] = []

                    hist_entry = {
                        "timestamp": st.session_state["active_time"],
                        "filename": audio_label,
                        "codec": selected_codec.upper(),
                        "mode": detection_mode,
                        "score": result["overall_liveness"],
                        "band": result["overall_band"],
                        "aasist_class": result.get("overall_aasist_class"),
                        "aasist_score": result.get("overall_aasist_bona_fide_score"),
                        "duration_sec": result["duration_sec"],
                        "result_obj": result,
                    }
                    st.session_state["analysis_history"].insert(0, hist_entry)
                    st.success("Analysis completed successfully.")
                except PipelineError as pe:
                    st.error(f"Analysis Pipeline Error: {pe}")
                except Exception as ex:
                    st.error(f"Unexpected Processing Error: {ex}")

        # =====================================================================
        # RESULTS PRESENTATION (COMPACT & BALANCED)
        # =====================================================================
        if "active_result" in st.session_state:
            res = st.session_state["active_result"]
            lbl = st.session_state.get("active_label", "Audio")
            score = res["overall_liveness"]
            band = res["overall_band"]
            active_mode = res.get("detection_mode", "pretrained")
            aasist_class = res.get("overall_aasist_class")

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #13A89E; margin-bottom: 5px;">'
                'DETECTION ASSESSMENT REPORT'
                '</div>',
                unsafe_allow_html=True,
            )

            # 1. Main Outcome Banner
            verdict_title, verdict_desc, verdict_css, verdict_color = format_cautious_outcome(score, aasist_class, active_mode)

            banner_html = f"""
            <div class="ss-verdict-banner {verdict_css}">
                <div>
                    <span class="ss-spec-pill" style="color: {verdict_color}; border-color: {verdict_color}; font-size: 0.66rem;">
                        ASSESSMENT OUTCOME
                    </span>
                    <h2 class="ss-verdict-title" style="color: {verdict_color}; margin-top: 4px;">
                        {verdict_title}
                    </h2>
                    <p class="ss-verdict-desc">{verdict_desc}</p>
                </div>
                <div style="text-align: right; min-width: 120px;">
                    <div style="font-size: 0.66rem; color: #707C8B; text-transform: uppercase; font-weight: 600;">Overall Score Index</div>
                    <div style="font-size: 1.75rem; font-weight: 800; font-family: 'JetBrains Mono'; color: {verdict_color}; line-height: 1.1;">
                        {score} <span style="font-size: 0.82rem; color: #707C8B;">/ 100</span>
                    </div>
                </div>
            </div>
            """
            st.markdown(banner_html, unsafe_allow_html=True)

            if res.get("warnings"):
                for w in res["warnings"]:
                    st.warning(f"⚠️ {w}")

            # 2. AASIST Primary Neural Engine Card
            aasist_loaded = res.get("aasist_model_loaded", False)
            aasist_score = res.get("overall_aasist_bona_fide_score")
            aasist_data = res.get("aasist", {})
            raw_logits = aasist_data.get("raw_score")

            aas_class_label = "Bona Fide-Like (Human)" if aasist_class == "bona_fide" else ("Spoof-Like (Synthetic)" if aasist_class == "spoof" else "Unavailable")
            aas_color = "#13A89E" if aasist_class == "bona_fide" else ("#D95757" if aasist_class == "spoof" else "#707C8B")

            st.markdown(
                f"""
                <div class="ss-card" style="border-left: 3px solid var(--ss-teal-accent);">
                    <div class="ss-card-header">
                        <div>
                            <span class="ss-engine-tag aasist">PRIMARY NEURAL ANTI-SPOOFING ENGINE</span>
                            <div class="ss-card-title">Pretrained AASIST Model (Clova AI / ICASSP 2022)</div>
                        </div>
                        <span class="ss-status-pill {'active' if aasist_loaded else 'inactive'}">
                            {'● Model Active' if aasist_loaded else '○ Unloaded'}
                        </span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 9px; margin-bottom: 6px;">
                        <div style="background: #F8FAF9; padding: 7px 10px; border-radius: 6px; border: 1px solid #E3E8ED;">
                            <div style="font-size: 0.66rem; color: #707C8B; text-transform: uppercase; font-weight: 600;">Predicted Neural Class</div>
                            <div style="font-size: 1rem; font-weight: 800; color: {aas_color}; margin: 2px 0;">{aas_class_label}</div>
                            <div style="font-size: 0.68rem; color: #8E97A4;">Maximum logit assignment</div>
                        </div>
                        <div style="background: #F8FAF9; padding: 7px 10px; border-radius: 6px; border: 1px solid #E3E8ED;">
                            <div style="font-size: 0.66rem; color: #707C8B; text-transform: uppercase; font-weight: 600;">Bona-Fide Model Score</div>
                            <div style="font-size: 1rem; font-weight: 800; font-family: 'JetBrains Mono'; color: #14283D; margin: 2px 0;">
                                {f"{aasist_score:.4f}" if aasist_score is not None else "N/A"} <span style="font-size: 0.72rem; color:#707C8B;">/ 1.0</span>
                            </div>
                            <div style="font-size: 0.68rem; color: #8E97A4;">Softmax evidence (uncalibrated)</div>
                        </div>
                        <div style="background: #F8FAF9; padding: 7px 10px; border-radius: 6px; border: 1px solid #E3E8ED;">
                            <div style="font-size: 0.66rem; color: #707C8B; text-transform: uppercase; font-weight: 600;">Raw Class Logits (z0, z1)</div>
                            <div style="font-size: 0.85rem; font-weight: 700; font-family: 'JetBrains Mono'; color: #14283D; margin: 2px 0;">
                                {f"Spoof: {raw_logits[0]:+.2f} | Bona: {raw_logits[1]:+.2f}" if raw_logits and len(raw_logits)==2 else "N/A"}
                            </div>
                            <div style="font-size: 0.68rem; color: #8E97A4;">Pre-softmax output logits</div>
                        </div>
                        <div style="background: #F8FAF9; padding: 7px 10px; border-radius: 6px; border: 1px solid #E3E8ED;">
                            <div style="font-size: 0.66rem; color: #707C8B; text-transform: uppercase; font-weight: 600;">Audio Processing Path</div>
                            <div style="font-size: 0.85rem; font-weight: 700; color: #14283D; margin: 2px 0;">
                                {res['codec'].upper()} ➔ 16 kHz Mono
                            </div>
                            <div style="font-size: 0.68rem; color: #8E97A4;">64,600 samples (~4.04s window)</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 3. Supporting Diagnostics: Physics & Biology Engine Cards
            col_diag_phys, col_diag_bio = st.columns(2)

            phys_val = int(round(res["physics_mean"] * 100))
            bio_val = int(round(res["biology_mean"] * 100))

            with col_diag_phys:
                st.markdown(
                    f"""
                    <div class="ss-card" style="border-left: 3px solid #0284C7;">
                        <div class="ss-card-header">
                            <div>
                                <span class="ss-engine-tag physics">Supporting Diagnostic</span>
                                <div class="ss-card-title">Physics Engine (Room & Echo)</div>
                            </div>
                            <span style="font-family: 'JetBrains Mono'; font-weight: 800; font-size: 1rem; color: #0284C7;">{phys_val} / 100</span>
                        </div>
                        <p style="font-size: 0.76rem; color: #707C8B; line-height: 1.4; margin: 3px 0 6px 0;">
                            Examines room reverberation decay (EDC), ambient pause noise floor, and spectral energy consistency.
                        </p>
                        <div style="font-size: 0.70rem; color: #8E97A4; border-top: 1px solid #EEF1F4; padding-top: 4px;">
                            Studio recordings with noise gates may yield lower reverberation scores.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_diag_bio:
                st.markdown(
                    f"""
                    <div class="ss-card" style="border-left: 3px solid #8B5CF6;">
                        <div class="ss-card-header">
                            <div>
                                <span class="ss-engine-tag biology">Supporting Diagnostic</span>
                                <div class="ss-card-title">Biology Engine (Vocal Dynamics)</div>
                            </div>
                            <span style="font-family: 'JetBrains Mono'; font-weight: 800; font-size: 1rem; color: #8B5CF6;">{bio_val} / 100</span>
                        </div>
                        <p style="font-size: 0.76rem; color: #707C8B; line-height: 1.4; margin: 3px 0 6px 0;">
                            Evaluates vocal tract stability, pitch jitter, micro-shimmer, and involuntary respiratory pauses.
                        </p>
                        <div style="font-size: 0.70rem; color: #8E97A4; border-top: 1px solid #EEF1F4; padding-top: 4px;">
                            Modern vocoders can synthesize realistic pitch contours.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 4. SELECTIVE DARK CONSOLE: Waveform & Audio Telemetry
            st.markdown(
                """
                <div class="ss-dark-console">
                    <div class="ss-card-header">
                        <div>
                            <div class="ss-card-title">AUDIO SIGNAL CONSOLE & TELEMETRY</div>
                            <div class="ss-card-subtitle">Waveform inspection, telephony codec playback, and rolling window metrics.</div>
                        </div>
                        <span class="ss-spec-pill" style="background:#14283D; border-color:#2A3F5B; color:#13A89E;">TELEMETRY ACTIVE</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            # Downsampled dark console waveform
            if "crushed_audio" in res:
                st.plotly_chart(render_dark_waveform(res["crushed_audio"], sr=8000), use_container_width=True, config={"displayModeBar": False})

            # Audio Players
            col_play1, col_play2 = st.columns(2)
            with col_play1:
                if "raw_audio" in res:
                    st.markdown("<div style='font-size:0.74rem; font-weight:600; color:#E2E8F0; margin-bottom:2px;'>Preprocessed Ingress (8 kHz Mono)</div>", unsafe_allow_html=True)
                    st.audio(audio_to_wav_bytes(res["raw_audio"], sr=8000), format="audio/wav")

            with col_play2:
                if "crushed_audio" in res:
                    st.markdown(f"<div style='font-size:0.74rem; font-weight:600; color:#E2E8F0; margin-bottom:2px;'>Telephone Codec Output ({res['codec'].upper()})</div>", unsafe_allow_html=True)
                    st.audio(audio_to_wav_bytes(res["crushed_audio"], sr=8000), format="audio/wav")

            st.markdown("</div>", unsafe_allow_html=True)

            # 5. Multi-Engine Minimalist Gauges & Rolling Window Timeline
            with st.container(border=True):
                st.markdown(
                    """
                    <div style="margin-bottom: 6px;">
                        <div style="font-size: 0.92rem; font-weight: 700; color: #14283D;">Engine Score Array & Rolling Timeline</div>
                        <div style="font-size: 0.74rem; color: #707C8B;">Individual uncalibrated model scores across 6s rolling windows.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_g1, col_g2, col_g3, col_g4 = st.columns(4)
                with col_g1:
                    p_band = "green" if phys_val >= 70 else ("amber" if phys_val >= 40 else "red")
                    st.plotly_chart(render_merkur_gauge(phys_val, p_band, "Physics", "Acoustic Room"), use_container_width=True, config={"displayModeBar": False})
                with col_g2:
                    b_band = "green" if bio_val >= 70 else ("amber" if bio_val >= 40 else "red")
                    st.plotly_chart(render_merkur_gauge(bio_val, b_band, "Biology", "Vocal Tract"), use_container_width=True, config={"displayModeBar": False})
                with col_g3:
                    aas_val = res.get("overall_aasist_liveness", 50)
                    a_band = "green" if aas_val >= 70 else ("amber" if aas_val >= 40 else "red")
                    st.plotly_chart(render_merkur_gauge(aas_val, a_band, "AASIST", "Neural Network"), use_container_width=True, config={"displayModeBar": False})
                with col_g4:
                    base_score = res["overall_baseline_liveness"]
                    base_band = res["overall_band"]
                    st.plotly_chart(render_merkur_gauge(base_score, base_band, "Baseline", "Physics + Biology"), use_container_width=True, config={"displayModeBar": False})

                # Rolling Window Timeline
                if res.get("windows"):
                    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
                    st.plotly_chart(render_timeline_chart(res["windows"], show_ml=False, show_aasist=True), use_container_width=True, config={"displayModeBar": False})

            # 6. Understanding the Detection Signals
            with st.container(border=True):
                st.markdown(
                    """
                    <div style="margin-bottom: 6px;">
                        <div style="font-size: 0.92rem; font-weight: 700; color: #14283D;">Signal Interpretation & Agent Guidance</div>
                        <div style="font-size: 0.74rem; color: #707C8B;">Synthesis of acoustic, biological, and neural indicators for human operators.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_sig1, col_sig2 = st.columns([1.5, 1.2])
                with col_sig1:
                    st.markdown("<div style='font-size:0.78rem; font-weight:700; color:#14283D; margin-bottom:4px;'>Primary Diagnostic Observations:</div>", unsafe_allow_html=True)
                    for r in res.get("reasons", []):
                        st.markdown(f"• **{r}**")

                with col_sig2:
                    if res.get("agent_tip"):
                        st.markdown(
                            f"""
                            <div style="background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 6px; padding: 8px 10px;">
                                <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: #D97706; margin-bottom: 2px;">
                                    🛡️ Suggested Agent Verification Prompt
                                </div>
                                <div style="font-size: 0.80rem; font-style: italic; color: #92400E; line-height: 1.4;">
                                    "{res['agent_tip']}"
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            """
                            <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 6px; padding: 8px 10px;">
                                <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: #15803D; margin-bottom: 2px;">
                                    Normal Verification Status
                                </div>
                                <div style="font-size: 0.76rem; color: #166534;">
                                    Signals fall within standard conversational parameters.
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        else:
            st.info("Upload an audio file or select a Demo Benchmark Sample above, then click Run Audio Analysis.")

    # =========================================================================
    # TAB 2: DUAL RECORDING COMPARISON
    # =========================================================================
    with compare_tab:
        with st.container(border=True):
            st.markdown(
                """
                <div style="margin-bottom: 8px;">
                    <div style="font-size: 0.95rem; font-weight: 700; color: #14283D;">Side-by-Side Audio Comparison</div>
                    <div style="font-size: 0.76rem; color: #707C8B;">Compare two audio recordings (e.g. reference speaker vs. voice clone) through the same telephony pipeline.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_cmp1, col_cmp2 = st.columns(2)
            with col_cmp1:
                st.markdown("<div style='font-size:0.84rem; font-weight:700; color:#14283D;'>Recording A (Reference Human)</div>", unsafe_allow_html=True)
                file_a = st.file_uploader("Upload Audio A (WAV)", type=["wav"], key="cmp_file_a_v3")
                smoke_a = st.selectbox("Or choose demo sample A", list(smoke_options.keys()), key="cmp_smoke_a_v3", index=1 if len(smoke_options) > 1 else 0)

            with col_cmp2:
                st.markdown("<div style='font-size:0.84rem; font-weight:700; color:#14283D;'>Recording B (Suspected Clone)</div>", unsafe_allow_html=True)
                file_b = st.file_uploader("Upload Audio B (WAV)", type=["wav"], key="cmp_file_b_v3")
                smoke_b = st.selectbox("Or choose demo sample B", list(smoke_options.keys()), key="cmp_smoke_b_v3", index=2 if len(smoke_options) > 2 else 0)

            cmp_codec = st.selectbox("Comparison Codec", ["g711", "clean"], key="cmp_codec_v3")
            cmp_run = st.button("Run Comparative Analysis", icon=":material/compare_arrows:", type="primary")

            src_a = file_a.getvalue() if file_a else (smoke_options[smoke_a] if smoke_a != "None (Upload file)" else None)
            src_b = file_b.getvalue() if file_b else (smoke_options[smoke_b] if smoke_b != "None (Upload file)" else None)

            if cmp_run and src_a and src_b:
                with st.spinner("Analyzing both recordings through identical codec pipelines..."):
                    try:
                        res_a = run_pipeline(src_a, codec=cmp_codec, detection_mode="compare")
                        res_b = run_pipeline(src_b, codec=cmp_codec, detection_mode="compare")

                        col_ra, col_rb = st.columns(2)
                        with col_ra:
                            st.markdown(f"### Recording A: **{res_a['overall_liveness']} / 100**")
                            st.plotly_chart(render_merkur_gauge(res_a["overall_liveness"], res_a["overall_band"], "Recording A"), use_container_width=True)
                            st.markdown(f"• Baseline Score: `{res_a['overall_baseline_liveness']}`")
                            st.markdown(f"• Pretrained AASIST: `{res_a.get('overall_aasist_class', 'N/A')}` (`{res_a.get('overall_aasist_bona_fide_score')}`)")
                            st.markdown(f"• Physics: `{int(res_a['physics_mean']*100)}%` | Biology: `{int(res_a['biology_mean']*100)}%`")

                        with col_rb:
                            st.markdown(f"### Recording B: **{res_b['overall_liveness']} / 100**")
                            st.plotly_chart(render_merkur_gauge(res_b["overall_liveness"], res_b["overall_band"], "Recording B"), use_container_width=True)
                            st.markdown(f"• Baseline Score: `{res_b['overall_baseline_liveness']}`")
                            st.markdown(f"• Pretrained AASIST: `{res_b.get('overall_aasist_class', 'N/A')}` (`{res_b.get('overall_aasist_bona_fide_score')}`)")
                            st.markdown(f"• Physics: `{int(res_b['physics_mean']*100)}%` | Biology: `{int(res_b['biology_mean']*100)}%`")

                    except Exception as ce:
                        st.error(f"Comparison Error: {ce}")
