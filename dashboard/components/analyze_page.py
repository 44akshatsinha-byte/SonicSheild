"""SonicShield Audio Analysis Page Component.

Prioritizes:
1. Dual Audio Ingress Modes: Mode A (File Upload) & Mode B (Live Microphone Recording)
2. Audio signal metadata grid with validation diagnostics
3. Cohesive analysis controls panel (telephony codecs, detection modes, impairments)
4. Safe temporary file handling with guaranteed cleanup for microphone recordings
5. Main detection result with cautious labels
6. Pretrained AASIST neural card showing both Bona-Fide and Spoof model scores
7. Supporting Physics & Biology diagnostic cards
8. Multi-engine consensus explanation ("What did SonicShield detect?")
9. Audio waveform in dark technical console & audio players
10. Side-by-side comparison mode
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
from src.audio_utils import validate_recording_bytes, safe_temporary_wav
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

def generate_multi_engine_explanation(res: dict) -> dict:
    """Generates an honest, multi-perspective explanation of what SonicShield detected.
    
    Synthesizes AASIST neural output, Physics Engine acoustic metrics, 
    and Biology Engine vocal perturbation to explain agreement or divergence.
    """
    aasist_class = res.get("overall_aasist_class")
    aasist_score = res.get("overall_aasist_bona_fide_score")
    phys_val = int(round(res.get("physics_mean", 0.5) * 100))
    bio_val = int(round(res.get("biology_mean", 0.5) * 100))

    # 1. Neural perspective
    if aasist_class == "bona_fide":
        neural_desc = (
            f"Pretrained AASIST neural graph attention observed spectro-temporal continuity characteristic "
            f"of authentic human speech (Bona-Fide Model Score: {aasist_score:.4f})."
            if aasist_score is not None else "AASIST leans bona-fide based on neural feature attention."
        )
    elif aasist_class == "spoof":
        neural_desc = (
            f"Pretrained AASIST neural graph attention flagged phase or spectro-temporal discontinuities "
            f"characteristic of synthetic vocoders or voice clones (Bona-Fide Model Score: {aasist_score:.4f})."
            if aasist_score is not None else "AASIST flagged synthetic or cloned speech patterns."
        )
    else:
        neural_desc = "AASIST neural inference was unavailable or inconclusive for this recording."

    # 2. Physics perspective
    if phys_val >= 60:
        phys_desc = f"Physics Engine recorded natural room acoustic reverberation decay (EDC) and expected ambient background pause tone ({phys_val}/100)."
    elif phys_val <= 40:
        phys_desc = f"Physics Engine observed atypical room acoustic dynamics or rapid unnatural energy drop-offs indicative of dry vocoder rendering or noise suppression ({phys_val}/100)."
    else:
        phys_desc = f"Physics Engine measured intermediate reverberation characteristics within typical office/headset bounds ({phys_val}/100)."

    # 3. Biology perspective
    if bio_val >= 60:
        bio_desc = f"Biology Engine measured organic glottal micro-perturbations (pitch jitter, shimmer) and natural respiratory intervals ({bio_val}/100)."
    elif bio_val <= 40:
        bio_desc = f"Biology Engine detected mathematical regularity or unnatural harmonic stability in pitch and glottal cycles ({bio_val}/100)."
    else:
        bio_desc = f"Biology Engine recorded moderate vocal stability consistent with conversational telephone speech ({bio_val}/100)."

    # 4. Engine agreement synthesis
    codec = res.get("codec", "clean")
    if aasist_class == "bona_fide" and phys_val >= 50 and bio_val >= 50:
        agreement = "Consensus: Acoustic physics, vocal tract dynamics, and deep neural graph representations all align towards authentic human speech patterns."
        agree_type = "aligned_human"
    elif aasist_class == "spoof" and (phys_val < 50 or bio_val < 50):
        agreement = "Consensus: Neural artifacts and supporting acoustic/vocal diagnostics consistently point towards synthetic or cloned voice attributes."
        agree_type = "aligned_synthetic"
    elif aasist_class == "spoof" and codec in ("g711", "amr") and phys_val >= 50 and bio_val >= 50:
        agreement = (
            "Divergent / Telephony Channel Effect: AASIST flagged spoof-like patterns, which is an expected consequence of 8 kHz telephone compression "
            "(frequencies above 3.4 kHz are physically truncated by the PSTN bandpass filter). In contrast, Physics and Biology engines confirmed natural room "
            "reverberation decay and organic vocal micro-perturbations. To test uncompressed speech without telephone artifacts, select 'Clean (16 kHz Native)'."
        )
        agree_type = "divergent"
    else:
        agreement = "Divergent Indicators: The neural anti-spoofing model and acoustic physical diagnostics exhibit differing evidence levels. Such divergence is common with high-fidelity vocoders in physical rooms or human speakers using noise-cancelling microphones. Human operator verification is recommended."
        agree_type = "divergent"

    return {
        "neural": neural_desc,
        "physics": phys_desc,
        "biology": bio_desc,
        "agreement": agreement,
        "agree_type": agree_type,
    }

def render_analyze_page():
    """Renders the comprehensive 4-step guided analysis workflow supporting uploads and live microphone recordings."""
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
            # Step 1: Audio Ingress & Input with Segmented Control (Upload vs Live Microphone)
            with st.container(border=True):
                col_h1, col_h2 = st.columns([2.5, 1.5])
                with col_h1:
                    st.markdown(
                        """
                        <div style="font-size: 0.92rem; font-weight: 700; color: #14283D; margin-bottom: 2px;">
                            1. Audio Ingress & Input
                        </div>
                        <div style="font-size: 0.74rem; color: #707C8B; margin-bottom: 8px;">
                            Choose an ingress mode: upload a WAV file or record your live voice directly.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col_h2:
                    st.markdown(
                        '<div style="text-align: right;"><span class="ss-spec-pill">Step 1</span></div>',
                        unsafe_allow_html=True,
                    )

                # Two distinct ingress modes
                ingress_options = ["📁 Upload Audio File", "🎙️ Record Live Voice"]
                saved_mode = st.session_state.get("ingress_mode", "📁 Upload Audio File")
                if saved_mode not in ingress_options:
                    saved_mode = "📁 Upload Audio File"

                ingress_mode = st.segmented_control(
                    "Input Source Mode",
                    options=ingress_options,
                    default=saved_mode,
                    key="analysis_ingress_selector",
                    label_visibility="collapsed",
                )

                audio_data = None
                audio_label = None
                audio_source_type = "Uploaded File"
                live_mic_metadata = None

                # =============================================================
                # MODE A: UPLOAD AUDIO FILE
                # =============================================================
                if ingress_mode == "📁 Upload Audio File":
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

                    if uploaded_file is not None:
                        audio_data = uploaded_file.getvalue()
                        audio_label = uploaded_file.name
                        audio_source_type = "Uploaded File"
                        if "queued_source" in st.session_state:
                            del st.session_state["queued_source"]
                    elif selected_smoke != "None (Upload file)" and smoke_options[selected_smoke]:
                        audio_data = smoke_options[selected_smoke]
                        audio_label = selected_smoke
                        audio_source_type = "Uploaded File"
                        if "queued_source" in st.session_state:
                            del st.session_state["queued_source"]
                    elif default_audio_source is not None:
                        audio_data = default_audio_source
                        audio_label = default_label
                        audio_source_type = "Uploaded File"

                # =============================================================
                # MODE B: RECORD LIVE VOICE VIA MICROPHONE
                # =============================================================
                else:
                    st.markdown(
                        """
                        <div class="ss-mic-hero">
                            <div class="ss-mic-center">
                                <div class="ss-mic-icon-circle">
                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                                        <line x1="12" y1="19" x2="12" y2="23"/>
                                        <line x1="8" y1="23" x2="16" y2="23"/>
                                    </svg>
                                </div>
                                <div style="font-size: 0.96rem; font-weight: 800; color: #14283D; line-height: 1.2;">
                                    Record Your Voice
                                </div>
                                <div style="font-size: 0.76rem; color: #707C8B; max-width: 380px; margin-top: 3px;">
                                    Speak naturally into your microphone. SonicShield will analyze your recording 
                                    using its acoustic, biological and neural detection engines.
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Streamlit Native Microphone Audio Input Widget
                    recorded_audio = st.audio_input(
                        "Record your voice (Speak for approximately 6–10 seconds)",
                        sample_rate=16000,
                        key="sonicshield_microphone_input",
                    )

                    if recorded_audio is None:
                        # State: Ready to Record & Instructions
                        st.markdown(
                            """
                            <div class="ss-instruction-list">
                                <strong style="color: #14283D; font-size: 0.74rem;">Ready to Record — Quick Instructions:</strong>
                                <ol>
                                    <li>Click the red record button in the recorder above to start.</li>
                                    <li>Allow browser microphone access when prompted.</li>
                                    <li>Keep your microphone reasonably close and speak in a quiet room.</li>
                                    <li>Speak naturally for approximately <strong>6–10 seconds</strong> (avoid long silence).</li>
                                    <li>Click stop when finished to preview your voice before analysis.</li>
                                </ol>
                                <div class="ss-privacy-badge">
                                    <span>🔒</span>
                                    <span><strong>Privacy Notice:</strong> Your recording is processed in memory by SonicShield. It is never saved permanently by default.</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        # State: Recording Complete & Validated
                        rec_bytes = recorded_audio.getvalue()
                        meta, mic_warnings, mic_error = validate_recording_bytes(rec_bytes)

                        if mic_error:
                            st.error(f"⚠️ Recording Error: {mic_error}")
                        else:
                            live_mic_metadata = meta
                            audio_data = rec_bytes
                            audio_label = f"Mic Recording ({meta['duration_sec']:.1f}s)"
                            audio_source_type = "Microphone Recording"

                            st.markdown(
                                f"""
                                <div style="display: flex; justify-content: space-between; align-items: center; margin: 6px 0 4px 0;">
                                    <span style="font-size: 0.78rem; font-weight: 700; color: #13A89E;">
                                        ● Recording Complete ({meta['duration_sec']:.2f}s)
                                    </span>
                                    <span class="ss-spec-pill" style="color: #13A89E; border-color: rgba(19, 168, 158, 0.4);">
                                        Ready to Analyze
                                    </span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            for w in mic_warnings:
                                st.warning(f"ℹ️ {w}")

                            # Show pre-analysis waveform preview from recorded audio
                            try:
                                wave_preview, preview_sr = sf.read(io.BytesIO(rec_bytes), dtype="float32")
                                if wave_preview.ndim > 1:
                                    wave_preview = np.mean(wave_preview, axis=1)
                                st.plotly_chart(
                                    render_dark_waveform(wave_preview, sr=preview_sr, title="RECORDED VOICE WAVEFORM (PREVIEW)"),
                                    use_container_width=True,
                                    config={"displayModeBar": False},
                                )
                            except Exception:
                                pass

                            st.markdown(
                                "<div style='font-size: 0.70rem; color: #8E97A4; margin-top: 2px;'>"
                                "💡 Tip: To re-record, delete the recording in the audio control above and speak again."
                                "</div>",
                                unsafe_allow_html=True,
                            )

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
                            Header diagnostics, channel specifications, and physical signal properties.
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
                        src_icon = "🎙️" if audio_source_type == "Microphone Recording" else "📁"

                        meta_html = f"""
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; font-size: 0.76rem;">
                            <div style="background: #F8FAF9; padding: 6px 9px; border-radius: 6px; border: 1px solid #E3E8ED;">
                                <span style="color: #707C8B;">Input Ingress Source:</span><br>
                                <strong style="color: #14283D;">{src_icon} {audio_label}</strong>
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
                                <span style="color: #707C8B;">Payload Size:</span><br>
                                <strong style="font-family: 'JetBrains Mono';">{fsize / 1024:.1f} KB</strong>
                            </div>
                        </div>
                        """
                        st.markdown(meta_html, unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Could not parse WAV headers: {e}")
                else:
                    if ingress_mode == "🎙️ Record Live Voice":
                        st.markdown(
                            "<p style='font-size:0.78rem; color:#707C8B; margin:0;'>Record your speech using the microphone control above to inspect physical signal telemetry.</p>",
                            unsafe_allow_html=True,
                        )
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

                codec_choices = ["clean", "g711"]
                if supported_codecs.get("amr"):
                    codec_choices.append("amr")

                selected_codec = st.selectbox(
                    "Telephony Codec Simulation",
                    options=codec_choices,
                    format_func=lambda x: {
                        "clean": "Clean (16 kHz Native / Direct Wideband Speech)",
                        "g711": "G.711 μ-law (8 kHz Standard PSTN Telephony)",
                        "amr": "AMR-NB 12.2k (8 kHz GSM Cellular)",
                    }.get(x, x),
                    index=0,
                )

                if selected_codec == "clean":
                    st.markdown(
                        """
                        <div style="font-size: 0.72rem; color: #0F766E; background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 5px; padding: 5px 8px; margin: 4px 0 8px 0; line-height: 1.35;">
                            <strong>Branch B (AASIST):</strong> Evaluating native 16 kHz wideband audio. Full frequency bandwidth (0–8,000 Hz) is preserved for deep spectro-temporal graph attention.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div style="font-size: 0.72rem; color: #9A3412; background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 5px; padding: 5px 8px; margin: 4px 0 8px 0; line-height: 1.35;">
                            <strong>Branch B (AASIST):</strong> Evaluating 8 kHz telephone-band audio upsampled to 16 kHz. Because telephone channels truncate frequencies above 3.4 kHz, AASIST high-frequency filterbanks lack signal energy, naturally depressing bona-fide scores (out-of-domain telephone speech).
                        </div>
                        """,
                        unsafe_allow_html=True,
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

                btn_label = "Analyze Live Recording" if audio_source_type == "Microphone Recording" else "Run Audio Analysis"
                btn_icon = ":material/mic:" if audio_source_type == "Microphone Recording" else ":material/play_arrow:"

                run_btn = st.button(btn_label, icon=btn_icon, type="primary", use_container_width=True)

        # Trigger Processing — Safe execution via existing pipeline
        if run_btn and audio_data is not None:
            spinner_msg = (
                "Executing live microphone analysis across acoustic, biological, and neural engines..."
                if audio_source_type == "Microphone Recording"
                else "Executing telephony compression and multi-engine inspection..."
            )
            with st.spinner(spinner_msg):
                try:
                    # Execute via temporary WAV file for microphone audio, guaranteed cleanup
                    if audio_source_type == "Microphone Recording" and isinstance(audio_data, bytes):
                        with safe_temporary_wav(audio_data, prefix="sonicshield_mic_") as temp_wav_path:
                            result = run_pipeline(
                                file_source=temp_wav_path,
                                codec=selected_codec,
                                mode=analysis_mode,
                                detection_mode=detection_mode,
                                packet_loss_rate=pkt_loss,
                                line_noise_snr=noise_snr,
                            )
                    else:
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
                    st.session_state["active_source_type"] = audio_source_type
                    st.session_state["active_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

                    if "analysis_history" not in st.session_state:
                        st.session_state["analysis_history"] = []

                    hist_entry = {
                        "timestamp": st.session_state["active_time"],
                        "filename": audio_label,
                        "source": audio_source_type,
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
            active_source = st.session_state.get("active_source_type", "Audio Recording")
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
                        ASSESSMENT OUTCOME ({active_source.upper()})
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

            # Saturated / Clipping Hardware Advisory Banner
            clip_info = res.get("clipping", {})
            if clip_info.get("is_clipped"):
                st.markdown(
                    f"""
                    <div style="background: #FEF2F2; border: 1.5px solid #FCA5A5; border-radius: 6px; padding: 10px 14px; margin: 8px 0 10px 0;">
                        <div style="font-size: 0.74rem; font-weight: 800; color: #DC2626; text-transform: uppercase; margin-bottom: 3px;">
                            ⚠️ Severe Microphone Input Overdrive Detected (Hardware Clipping)
                        </div>
                        <div style="font-size: 0.78rem; color: #7F1D1D; line-height: 1.45;">
                            Your recording reached digital ceiling saturation (<strong>{clip_info.get('pct_999', 0):.2f}%</strong> saturated samples, 
                            with up to <strong>{clip_info.get('max_consecutive_rail', 0)} consecutive samples</strong> stuck at the rail). 
                            Flat-topped waveform clipping produces artificial odd-harmonic distortion that causes AASIST neural filterbanks to output synthetic spoof flags.
                            <br><span style="color: #991B1B; font-weight: 600;">Recommended Action:</span> Lower your microphone input gain in Windows/Mac Sound Settings to 60%–70% 
                            or speak 15–20 cm back from the microphone to restore unclipped acoustic dynamics.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if res.get("warnings"):
                for w in res["warnings"]:
                    st.warning(f"⚠️ {w}")

            # 2. AASIST Primary Neural Engine Card (Displaying both Bona-Fide & Spoof Model Scores)
            aasist_loaded = res.get("aasist_model_loaded", False)
            aasist_bona_score = res.get("overall_aasist_bona_fide_score")
            aasist_spoof_score = round(1.0 - aasist_bona_score, 4) if aasist_bona_score is not None else None
            aasist_data = res.get("aasist", {})
            raw_logits = aasist_data.get("raw_score")

            aas_class_label = (
                "Bona Fide-Like (Human)" if aasist_class == "bona_fide"
                else ("Spoof-Like (Synthetic)" if aasist_class == "spoof" else "Unavailable")
            )
            aas_color = "#13A89E" if aasist_class == "bona_fide" else ("#D95757" if aasist_class == "spoof" else "#707C8B")

            if res.get("codec") == "clean":
                ingress_path_html = f"{active_source.split()[0]} ➔ Direct 16 kHz Wideband"
                ingress_sub_html = "Native wideband (0–8 kHz preserved)"
            else:
                ingress_path_html = f"{active_source.split()[0]} ➔ 8k {res['codec'].upper()} ➔ 16k Polyphase"
                ingress_sub_html = "Telephone bandlimited (<3.4 kHz)"

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
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 9px; margin-bottom: 6px;">
                        <div style="background: #F8FAF9; padding: 7px 10px; border-radius: 6px; border: 1px solid #E3E8ED;">
                            <div style="font-size: 0.66rem; color: #707C8B; text-transform: uppercase; font-weight: 600;">Predicted Neural Class</div>
                            <div style="font-size: 0.98rem; font-weight: 800; color: {aas_color}; margin: 2px 0;">{aas_class_label}</div>
                            <div style="font-size: 0.68rem; color: #8E97A4;">Maximum logit assignment</div>
                        </div>
                        <div style="background: #F8FAF9; padding: 7px 10px; border-radius: 6px; border: 1px solid #E3E8ED;">
                            <div style="font-size: 0.66rem; color: #707C8B; text-transform: uppercase; font-weight: 600;">AASIST Bona-Fide Score</div>
                            <div style="font-size: 0.98rem; font-weight: 800; font-family: 'JetBrains Mono'; color: #13A89E; margin: 2px 0;">
                                {f"{aasist_bona_score:.4f}" if aasist_bona_score is not None else "N/A"} <span style="font-size: 0.72rem; color:#707C8B;">/ 1.0</span>
                            </div>
                            <div style="font-size: 0.68rem; color: #8E97A4;">Softmax output (uncalibrated)</div>
                        </div>
                        <div style="background: #F8FAF9; padding: 7px 10px; border-radius: 6px; border: 1px solid #E3E8ED;">
                            <div style="font-size: 0.66rem; color: #707C8B; text-transform: uppercase; font-weight: 600;">AASIST Spoof Score</div>
                            <div style="font-size: 0.98rem; font-weight: 800; font-family: 'JetBrains Mono'; color: #D95757; margin: 2px 0;">
                                {f"{aasist_spoof_score:.4f}" if aasist_spoof_score is not None else "N/A"} <span style="font-size: 0.72rem; color:#707C8B;">/ 1.0</span>
                            </div>
                            <div style="font-size: 0.68rem; color: #8E97A4;">Softmax output (uncalibrated)</div>
                        </div>
                        <div style="background: #F8FAF9; padding: 7px 10px; border-radius: 6px; border: 1px solid #E3E8ED;">
                            <div style="font-size: 0.66rem; color: #707C8B; text-transform: uppercase; font-weight: 600;">Raw Class Logits (z0, z1)</div>
                            <div style="font-size: 0.82rem; font-weight: 700; font-family: 'JetBrains Mono'; color: #14283D; margin: 2px 0;">
                                {f"Spoof: {raw_logits[0]:+.2f} | Bona: {raw_logits[1]:+.2f}" if raw_logits and len(raw_logits)==2 else "N/A"}
                            </div>
                            <div style="font-size: 0.68rem; color: #8E97A4;">Pre-softmax output logits</div>
                        </div>
                        <div style="background: #F8FAF9; padding: 7px 10px; border-radius: 6px; border: 1px solid #E3E8ED;">
                            <div style="font-size: 0.66rem; color: #707C8B; text-transform: uppercase; font-weight: 600;">Ingress & Codec Path</div>
                            <div style="font-size: 0.82rem; font-weight: 700; color: #14283D; margin: 2px 0;">
                                {ingress_path_html}
                            </div>
                            <div style="font-size: 0.68rem; color: #8E97A4;">{ingress_sub_html}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if res.get("codec") == "clean":
                st.markdown(
                    """
                    <div style="font-size: 0.73rem; color: #0F766E; background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 6px; padding: 7px 11px; margin-top: 4px; line-height: 1.4;">
                        🛡️ <strong>Independent Dual-Branch Pipeline:</strong> AASIST evaluated native 16 kHz wideband speech preserving full acoustic frequency resolution (0–8,000 Hz). In parallel, Branch A evaluated 8 kHz telephonic audio for Room Acoustics (Physics Engine) and Vocal Perturbations (Biology Engine).
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div style="font-size: 0.73rem; color: #9A3412; background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 6px; padding: 7px 11px; margin-top: 4px; line-height: 1.4;">
                        ⚠️ <strong>Out-of-Domain Telephone Bandwidth Effect:</strong> AASIST evaluated 8 kHz telephone-band audio upsampled to 16 kHz. Standard PSTN / cellular codecs filter out frequencies above 3.4 kHz. Because AASIST was trained on wideband speech with active acoustic energy up to 8 kHz, telephone audio naturally receives depressed bona-fide scores due to channel bandlimiting. To inspect uncompressed microphone speech, select <strong>Clean (16 kHz Native)</strong>.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Technical Pre-Inference Telemetry Expander (Phase 3 Audit)
            aas_diag = aasist_data.get("diagnostics")
            if aas_diag:
                with st.expander("🔬 AASIST Pre-Inference Waveform Diagnostics (Exact 64,600 Samples Audited)"):
                    diag_cols = st.columns(4)
                    with diag_cols[0]:
                        st.markdown(
                            f"<div style='font-size:0.70rem;color:#707C8B;'>Sampling Rate & Window</div>"
                            f"<div style='font-size:0.80rem;font-weight:700;font-family:JetBrains Mono;'>{aas_diag.get('target_sample_rate')} Hz ({aas_diag.get('fed_num_samples')} samples)</div>"
                            f"<div style='font-size:0.68rem;color:#8E97A4;'>Exact 4.038s evaluation standard</div>",
                            unsafe_allow_html=True,
                        )
                    with diag_cols[1]:
                        st.markdown(
                            f"<div style='font-size:0.70rem;color:#707C8B;'>Amplitude & RMS Energy</div>"
                            f"<div style='font-size:0.80rem;font-weight:700;font-family:JetBrains Mono;'>Max: {aas_diag.get('abs_max_amplitude')} | RMS: {aas_diag.get('rms_energy')}</div>"
                            f"<div style='font-size:0.68rem;color:#8E97A4;'>Range [{aas_diag.get('min_amplitude')}, {aas_diag.get('max_amplitude')}]</div>",
                            unsafe_allow_html=True,
                        )
                    with diag_cols[2]:
                        sat_color = "#DC2626" if aas_diag.get('saturated_samples_pct', 0) > 0.02 else "#16A6A0"
                        st.markdown(
                            f"<div style='font-size:0.70rem;color:#707C8B;'>Saturation & Distortion</div>"
                            f"<div style='font-size:0.80rem;font-weight:700;font-family:JetBrains Mono;color:{sat_color};'>{aas_diag.get('saturated_samples_pct')}% at rail</div>"
                            f"<div style='font-size:0.68rem;color:#8E97A4;'>{aas_diag.get('saturated_samples_count')} samples >= 0.999</div>",
                            unsafe_allow_html=True,
                        )
                    with diag_cols[3]:
                        st.markdown(
                            f"<div style='font-size:0.70rem;color:#707C8B;'>PyTorch Execution State</div>"
                            f"<div style='font-size:0.80rem;font-weight:700;font-family:JetBrains Mono;color:#13A89E;'>eval() + inference_mode</div>"
                            f"<div style='font-size:0.68rem;color:#8E97A4;'>Non-finite count: {aas_diag.get('non_finite_count')}</div>",
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
                            Close-talk headset microphones or noise-cancelling hardware naturally produce lower reverberation tails.
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
                            Advanced neural vocoders can synthesize pitch micro-perturbations; treat as supporting evidence.
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
                if res.get("codec") == "clean" and "audio_16k" in res:
                    st.markdown("<div style='font-size:0.74rem; font-weight:600; color:#E2E8F0; margin-bottom:2px;'>Branch B AASIST Ingress (16 kHz Wideband Mono)</div>", unsafe_allow_html=True)
                    st.audio(audio_to_wav_bytes(res["audio_16k"], sr=16000), format="audio/wav")
                elif "raw_audio" in res:
                    st.markdown("<div style='font-size:0.74rem; font-weight:600; color:#E2E8F0; margin-bottom:2px;'>Branch A Ingress (8 kHz Telephony Mono)</div>", unsafe_allow_html=True)
                    st.audio(audio_to_wav_bytes(res["raw_audio"], sr=8000), format="audio/wav")

            with col_play2:
                if res.get("codec") == "clean" and "raw_audio" in res:
                    st.markdown("<div style='font-size:0.74rem; font-weight:600; color:#E2E8F0; margin-bottom:2px;'>Branch A Diagnostics Ingress (8 kHz Mono)</div>", unsafe_allow_html=True)
                    st.audio(audio_to_wav_bytes(res["raw_audio"], sr=8000), format="audio/wav")
                elif "crushed_audio" in res:
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

            # 6. Understanding the Detection Signals: What Did SonicShield Detect?
            explanation = generate_multi_engine_explanation(res)
            with st.container(border=True):
                st.markdown(
                    """
                    <div style="margin-bottom: 8px;">
                        <div style="font-size: 0.95rem; font-weight: 700; color: #14283D;">What did SonicShield detect?</div>
                        <div style="font-size: 0.76rem; color: #707C8B;">Coordinated multi-perspective synthesis explaining acoustic, biological, and neural indicators.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_exp1, col_exp2, col_exp3 = st.columns(3)
                with col_exp1:
                    st.markdown(
                        f"""
                        <div style="background: #F8FAF9; border: 1px solid #E3E8ED; border-top: 2.5px solid var(--ss-teal-accent); border-radius: 6px; padding: 8px 10px; height: 100%;">
                            <div style="font-size: 0.70rem; font-weight: 700; color: #13A89E; text-transform: uppercase; margin-bottom: 3px;">1. Neural Representation</div>
                            <div style="font-size: 0.76rem; color: #17212B; line-height: 1.45;">{explanation['neural']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col_exp2:
                    st.markdown(
                        f"""
                        <div style="background: #F8FAF9; border: 1px solid #E3E8ED; border-top: 2.5px solid #0284C7; border-radius: 6px; padding: 8px 10px; height: 100%;">
                            <div style="font-size: 0.70rem; font-weight: 700; color: #0284C7; text-transform: uppercase; margin-bottom: 3px;">2. Acoustic Environment</div>
                            <div style="font-size: 0.76rem; color: #17212B; line-height: 1.45;">{explanation['physics']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col_exp3:
                    st.markdown(
                        f"""
                        <div style="background: #F8FAF9; border: 1px solid #E3E8ED; border-top: 2.5px solid #8B5CF6; border-radius: 6px; padding: 8px 10px; height: 100%;">
                            <div style="font-size: 0.70rem; font-weight: 700; color: #8B5CF6; text-transform: uppercase; margin-bottom: 3px;">3. Vocal Physiology</div>
                            <div style="font-size: 0.76rem; color: #17212B; line-height: 1.45;">{explanation['biology']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

                # Consensus or Divergence Box
                if explanation["agree_type"] == "divergent":
                    cons_bg, cons_border, cons_color = "#FFFBEB", "#FDE68A", "#D97706"
                elif explanation["agree_type"] == "aligned_human":
                    cons_bg, cons_border, cons_color = "#F0FDF4", "#BBF7D0", "#15803D"
                else:
                    cons_bg, cons_border, cons_color = "#FEF2F2", "#FECACA", "#DC2626"

                st.markdown(
                    f"""
                    <div style="background: {cons_bg}; border: 1px solid {cons_border}; border-radius: 6px; padding: 9px 12px; margin-bottom: 8px;">
                        <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: {cons_color}; margin-bottom: 2px;">
                            🛡️ Multi-Engine Consensus & Operator Advisory
                        </div>
                        <div style="font-size: 0.78rem; color: #1E293B; line-height: 1.45;">
                            {explanation['agreement']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if res.get("agent_tip"):
                    st.markdown(
                        f"""
                        <div style="background: #F8FAFC; border: 1px dashed #CBD5E1; border-radius: 6px; padding: 7px 11px;">
                            <span style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: #64748B;">Suggested Conversational Verification Prompt:</span>
                            <div style="font-size: 0.78rem; font-style: italic; color: #334155; margin-top: 2px;">
                                "{res['agent_tip']}"
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        else:
            st.info("Record your voice or upload a WAV file above, then click Run Audio Analysis to inspect multi-engine telemetry.")

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
