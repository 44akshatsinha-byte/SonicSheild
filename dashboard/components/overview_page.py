"""SonicShield Compact Overview Page Component.

Redesigned with professional proportions, compact hero with soundwave graphic,
3-column engine cards, unified audio upload card, and structured benchmark metrics.
"""
from __future__ import annotations

import os
import streamlit as st
from src.pretrained_aasist import get_aasist_classifier

def render_overview_page():
    """Renders the balanced, compact overview landing dashboard."""
    aasist = get_aasist_classifier()

    # 1. Compact Hero Section with Sound-Wave Graphic on Right
    st.markdown(
        """
        <div class="ss-hero-compact">
            <div style="display: flex; justify-content: space-between; align-items: center; gap: 20px; flex-wrap: wrap;">
                <div style="flex: 1 1 520px; min-width: 280px;">
                    <div class="ss-hero-badge">
                        <span>●</span>
                        <span>CYBERSECURITY AI AUDIO INTELLIGENCE</span>
                    </div>
                    <h1 class="ss-hero-h1">
                        Trust the Voice. Verify the Signal.
                    </h1>
                    <p class="ss-hero-lead-compact">
                        Advanced multi-engine audio intelligence for detecting synthetic voices, AI clones, 
                        and deepfake audio across acoustic, biological, and neural dimensions under realistic telephony conditions.
                    </p>
                </div>
                <!-- Distinctive Audio Waveform Graphic on Right -->
                <div style="flex: 0 0 280px; background: #0E1724; border: 1px solid rgba(255,255,255,0.12); border-radius: 8px; padding: 10px 14px; box-shadow: 0 2px 10px rgba(14,23,36,0.12);">
                    <div style="display: flex; justify-content: space-between; font-size: 0.66rem; color: #94A3B8; font-family: 'JetBrains Mono'; margin-bottom: 5px;">
                        <span>SPECTRO-TEMPORAL SIGNAL</span>
                        <span style="color: #13A89E; font-weight: 600;">16 kHz POLYPHASE</span>
                    </div>
                    <svg width="250" height="38" viewBox="0 0 250 38" fill="none" xmlns="http://www.w3.org/2000/svg" style="display: block;">
                        <!-- Acoustic Spectrum Bars -->
                        <rect x="0" y="14" width="3" height="10" rx="1.5" fill="#13A89E" opacity="0.4"/>
                        <rect x="6" y="10" width="3" height="18" rx="1.5" fill="#13A89E" opacity="0.6"/>
                        <rect x="12" y="6" width="3" height="26" rx="1.5" fill="#13A89E" opacity="0.75"/>
                        <rect x="18" y="12" width="3" height="14" rx="1.5" fill="#13A89E" opacity="0.5"/>
                        <rect x="24" y="2" width="3" height="34" rx="1.5" fill="#13A89E"/>
                        <rect x="30" y="8" width="3" height="22" rx="1.5" fill="#13A89E" opacity="0.7"/>
                        <rect x="36" y="13" width="3" height="12" rx="1.5" fill="#13A89E" opacity="0.4"/>
                        <rect x="42" y="5" width="3" height="28" rx="1.5" fill="#13A89E" opacity="0.85"/>
                        <rect x="48" y="11" width="3" height="16" rx="1.5" fill="#13A89E" opacity="0.6"/>
                        <rect x="54" y="1" width="3" height="36" rx="1.5" fill="#8AB8C5"/>
                        <rect x="60" y="7" width="3" height="24" rx="1.5" fill="#13A89E" opacity="0.8"/>
                        <rect x="66" y="14" width="3" height="10" rx="1.5" fill="#13A89E" opacity="0.4"/>
                        <rect x="72" y="3" width="3" height="32" rx="1.5" fill="#13A89E"/>
                        <rect x="78" y="9" width="3" height="20" rx="1.5" fill="#13A89E" opacity="0.7"/>
                        <rect x="84" y="12" width="3" height="14" rx="1.5" fill="#13A89E" opacity="0.5"/>
                        <rect x="90" y="2" width="3" height="34" rx="1.5" fill="#8AB8C5"/>
                        <rect x="96" y="6" width="3" height="26" rx="1.5" fill="#13A89E" opacity="0.85"/>
                        <rect x="102" y="13" width="3" height="12" rx="1.5" fill="#13A89E" opacity="0.4"/>
                        <rect x="108" y="4" width="3" height="30" rx="1.5" fill="#13A89E"/>
                        <rect x="114" y="10" width="3" height="18" rx="1.5" fill="#13A89E" opacity="0.65"/>
                        <rect x="120" y="15" width="3" height="8" rx="1.5" fill="#13A89E" opacity="0.3"/>
                        <rect x="126" y="2" width="3" height="34" rx="1.5" fill="#8AB8C5"/>
                        <rect x="132" y="8" width="3" height="22" rx="1.5" fill="#13A89E" opacity="0.7"/>
                        <rect x="138" y="12" width="3" height="14" rx="1.5" fill="#13A89E" opacity="0.5"/>
                        <rect x="144" y="5" width="3" height="28" rx="1.5" fill="#13A89E" opacity="0.85"/>
                        <rect x="150" y="11" width="3" height="16" rx="1.5" fill="#13A89E" opacity="0.6"/>
                        <rect x="156" y="1" width="3" height="36" rx="1.5" fill="#13A89E"/>
                        <rect x="162" y="7" width="3" height="24" rx="1.5" fill="#13A89E" opacity="0.8"/>
                        <rect x="168" y="14" width="3" height="10" rx="1.5" fill="#13A89E" opacity="0.4"/>
                        <rect x="174" y="3" width="3" height="32" rx="1.5" fill="#8AB8C5"/>
                        <rect x="180" y="9" width="3" height="20" rx="1.5" fill="#13A89E" opacity="0.7"/>
                        <rect x="186" y="13" width="3" height="12" rx="1.5" fill="#13A89E" opacity="0.45"/>
                        <rect x="192" y="1" width="3" height="36" rx="1.5" fill="#13A89E"/>
                        <rect x="198" y="6" width="3" height="26" rx="1.5" fill="#13A89E" opacity="0.8"/>
                        <rect x="204" y="12" width="3" height="14" rx="1.5" fill="#13A89E" opacity="0.5"/>
                        <rect x="210" y="4" width="3" height="30" rx="1.5" fill="#13A89E"/>
                        <rect x="216" y="10" width="3" height="18" rx="1.5" fill="#13A89E" opacity="0.6"/>
                        <rect x="222" y="15" width="3" height="8" rx="1.5" fill="#13A89E" opacity="0.35"/>
                        <rect x="228" y="3" width="3" height="32" rx="1.5" fill="#8AB8C5"/>
                        <rect x="234" y="8" width="3" height="22" rx="1.5" fill="#13A89E" opacity="0.7"/>
                        <rect x="240" y="13" width="3" height="12" rx="1.5" fill="#13A89E" opacity="0.4"/>
                        <rect x="246" y="6" width="3" height="26" rx="1.5" fill="#13A89E" opacity="0.75"/>
                    </svg>
                    <div style="display: flex; justify-content: space-between; font-size: 0.62rem; color: #707C8B; font-family: 'JetBrains Mono'; margin-top: 3px;">
                        <span>0 Hz</span>
                        <span>4 kHz (G.711)</span>
                        <span>8 kHz Nyquist</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hero Action Buttons
    col_cta1, col_cta2, col_cta3, col_spacer = st.columns([1.3, 1.5, 1.5, 1.7])
    with col_cta1:
        if st.button("Upload & Inspect", icon=":material/upload_file:", type="primary", use_container_width=True):
            st.session_state["ingress_mode"] = "📁 Upload Audio File"
            st.session_state["current_page"] = "Analyze Audio"
            st.rerun()
    with col_cta2:
        if st.button("Record Live Voice", icon=":material/mic:", type="secondary", use_container_width=True):
            st.session_state["ingress_mode"] = "🎙️ Record Live Voice"
            st.session_state["current_page"] = "Analyze Audio"
            st.rerun()
    with col_cta3:
        if st.button("Engine Architecture", icon=":material/insights:", type="secondary", use_container_width=True):
            st.session_state["current_page"] = "Engine Insights"
            st.rerun()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 2. Three Coordinated Engine Cards in 3-Column Layout
    st.markdown(
        '<div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #707C8B; margin-bottom: 7px;">'
        'THREE LAYERED ANALYSIS ENGINES'
        '</div>',
        unsafe_allow_html=True,
    )

    col_e1, col_e2, col_e3 = st.columns(3)

    with col_e1:
        card_phys_html = """
        <div class="ss-engine-card physics">
            <div>
                <span class="ss-engine-tag physics">Acoustic Physics</span>
                <div style="font-size: 0.95rem; font-weight: 700; color: #14283D; margin-bottom: 3px;">
                    Physics Engine
                </div>
                <p style="font-size: 0.78rem; color: #707C8B; line-height: 1.4; margin-bottom: 8px;">
                    Examines room reverberation decay (EDC), ambient pause tone, and acoustic reflection naturalness.
                </p>
                <div>
                    <span class="ss-feature-chip">Reverberation Tail</span>
                    <span class="ss-feature-chip">Pause Noise Floor</span>
                    <span class="ss-feature-chip">Echo Dynamics</span>
                </div>
            </div>
            <div style="margin-top: 12px; padding-top: 7px; border-top: 1px solid #EEF1F4; font-size: 0.68rem; color: #8E97A4;">
                Role: <strong>Supporting Acoustic Diagnostics</strong>
            </div>
        </div>
        """
        st.markdown(card_phys_html, unsafe_allow_html=True)

    with col_e2:
        card_bio_html = """
        <div class="ss-engine-card biology">
            <div>
                <span class="ss-engine-tag biology">Vocal Physiology</span>
                <div style="font-size: 0.95rem; font-weight: 700; color: #14283D; margin-bottom: 3px;">
                    Biology Engine
                </div>
                <p style="font-size: 0.78rem; color: #707C8B; line-height: 1.4; margin-bottom: 8px;">
                    Measures glottal micro-jitter, shimmer, pitch perturbation, and involuntary respiratory intervals.
                </p>
                <div>
                    <span class="ss-feature-chip">Pitch Perturbation</span>
                    <span class="ss-feature-chip">Jitter & Shimmer</span>
                    <span class="ss-feature-chip">Breath Intakes</span>
                </div>
            </div>
            <div style="margin-top: 12px; padding-top: 7px; border-top: 1px solid #EEF1F4; font-size: 0.68rem; color: #8E97A4;">
                Role: <strong>Supporting Vocal Diagnostics</strong>
            </div>
        </div>
        """
        st.markdown(card_bio_html, unsafe_allow_html=True)

    with col_e3:
        aasist_loaded = aasist.is_loaded
        card_aasist_html = f"""
        <div class="ss-engine-card aasist">
            <div>
                <span class="ss-engine-tag aasist">Deep Neural Model</span>
                <div style="font-size: 0.95rem; font-weight: 700; color: #14283D; margin-bottom: 3px;">
                    AASIST Neural Engine
                </div>
                <p style="font-size: 0.78rem; color: #707C8B; line-height: 1.4; margin-bottom: 8px;">
                    Pretrained spectro-temporal graph attention network on raw 16 kHz waveforms detecting cloning artifacts.
                </p>
                <div>
                    <span class="ss-feature-chip">Graph Attention</span>
                    <span class="ss-feature-chip">Bona-Fide Logits</span>
                    <span class="ss-feature-chip">Spoof Logits</span>
                </div>
            </div>
            <div style="margin-top: 12px; padding-top: 7px; border-top: 1px solid #EEF1F4; font-size: 0.68rem; color: #13A89E;">
                Role: <strong>Primary Neural Detection Engine</strong>
            </div>
        </div>
        """
        st.markdown(card_aasist_html, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 3. Single Integrated Quick Analysis Upload Card via st.container(border=True)
    with st.container(border=True):
        col_hdr1, col_hdr2 = st.columns([3, 1])
        with col_hdr1:
            st.markdown(
                """
                <div style="margin-bottom: 6px;">
                    <div style="font-size: 0.94rem; font-weight: 700; color: #14283D;">Quick Audio Inspection</div>
                    <div style="font-size: 0.76rem; color: #707C8B;">Upload a voice recording or test an included benchmark audio sample.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_hdr2:
            st.markdown(
                '<div style="text-align: right;"><span class="ss-spec-pill">Ready for Ingress</span></div>',
                unsafe_allow_html=True,
            )

        col_up, col_demo = st.columns([1.5, 1.1])

        smoke_dir = "data/smoke"
        smoke_options = {"None (Upload file)": None}
        if os.path.exists(smoke_dir):
            for f in sorted(os.listdir(smoke_dir)):
                if f.endswith(".wav"):
                    lbl = f.replace(".wav", "").replace("smoke_", "").replace("_", " ").title()
                    smoke_options[f"Demo: {lbl}"] = os.path.join(smoke_dir, f)

        with col_up:
            uploaded_overview = st.file_uploader(
                "Upload audio file (WAV, MP3, FLAC, OGG)",
                type=["wav", "mp3", "flac", "ogg"],
                key="overview_uploader_v3",
                help="Accepts WAV, MP3, FLAC, and OGG formats. File will be processed according to telephony codec.",
                label_visibility="collapsed",
            )

        with col_demo:
            st.markdown("<div style='font-size:0.76rem; font-weight:600; color:#14283D; margin-bottom:2px;'>Choose Preloaded Demo Sample:</div>", unsafe_allow_html=True)
            selected_sample = st.selectbox(
                "Preloaded Audio Samples",
                list(smoke_options.keys()),
                key="overview_demo_select_v3",
                label_visibility="collapsed",
            )

            if st.button("Run Audio Analysis", icon=":material/play_arrow:", type="primary", use_container_width=True):
                if uploaded_overview is not None:
                    st.session_state["queued_source"] = uploaded_overview.getvalue()
                    st.session_state["queued_name"] = uploaded_overview.name
                    st.session_state["current_page"] = "Analyze Audio"
                    st.rerun()
                elif selected_sample != "None (Upload file)" and smoke_options[selected_sample]:
                    st.session_state["queued_source"] = smoke_options[selected_sample]
                    st.session_state["queued_name"] = selected_sample
                    st.session_state["current_page"] = "Analyze Audio"
                    st.rerun()
                else:
                    st.warning("Please upload a file or select a demo sample first.")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 4. Structured Benchmark Summary Strip (Clean Grid, Never Clips)
    st.markdown(
        """
        <div class="ss-benchmark-grid">
            <div class="ss-bench-cell">
                <span class="ss-bench-label">Benchmark Cohort</span>
                <span class="ss-bench-val">200 Files</span>
                <span style="font-size: 0.68rem; color: #707C8B; margin-top: 2px;">In-The-Wild Release</span>
            </div>
            <div class="ss-bench-cell">
                <span class="ss-bench-label">Pretrained AASIST</span>
                <span class="ss-bench-val" style="color: #14283D;">56.5%</span>
                <span style="font-size: 0.68rem; color: #13A89E; font-weight: 600; margin-top: 2px;">ROC-AUC: 0.6225</span>
            </div>
            <div class="ss-bench-cell">
                <span class="ss-bench-label">Experimental Hybrid</span>
                <span class="ss-bench-val" style="color: #14283D;">57.5%</span>
                <span style="font-size: 0.68rem; color: #D99A35; font-weight: 600; margin-top: 2px;">ROC-AUC: 0.5712</span>
            </div>
            <div class="ss-bench-cell">
                <span class="ss-bench-label">Spoofs Caught</span>
                <span class="ss-bench-val" style="color: #14283D;">36 / 100</span>
                <span style="font-size: 0.68rem; color: #707C8B; margin-top: 2px;">Equal Detection Rate</span>
            </div>
        </div>
        <div style="background: #FFFFFF; border: 1px solid #E3E8ED; border-radius: 7px; padding: 7px 12px; font-size: 0.72rem; color: #707C8B; line-height: 1.4; margin-bottom: 0.85rem;">
            ⚠️ <strong>Research Scope:</strong> Exploratory benchmark under telephony domain shift. Standalone AASIST exhibits higher discriminative ranking than the uncalibrated linear hybrid. Scores represent raw model outputs, not calibrated probabilities.
        </div>
        """,
        unsafe_allow_html=True,
    )
