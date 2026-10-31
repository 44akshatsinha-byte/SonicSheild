"""SonicShield — AI Voice Clone Detection, Audio Liveness & Deepfake Audio Analysis.

Premium Cybersecurity SaaS Redesign inspired by Merkur Design.
Main entry point orchestrating modular components, state routing, and telephony pipelines.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="SonicShield | Acoustic AI Cybersecurity",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject Merkur-Inspired Design System CSS
import importlib
from dashboard.components import (
    styles,
    sidebar,
    header,
    overview_page,
    analyze_page,
    insights_page,
    evaluation_page,
    history_page,
)

importlib.reload(styles)
importlib.reload(sidebar)
importlib.reload(header)
importlib.reload(overview_page)
importlib.reload(analyze_page)
importlib.reload(insights_page)
importlib.reload(evaluation_page)
importlib.reload(history_page)

st.markdown(styles.CUSTOM_CSS, unsafe_allow_html=True)

# 3. Component aliases
render_sidebar = sidebar.render_sidebar
render_top_header = header.render_top_header
render_overview_page = overview_page.render_overview_page
render_analyze_page = analyze_page.render_analyze_page
render_insights_page = insights_page.render_insights_page
render_evaluation_page = evaluation_page.render_evaluation_page
render_history_page = history_page.render_history_page

# 4. Initialize Session State
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Overview"

if "analysis_history" not in st.session_state:
    st.session_state["analysis_history"] = []

# 5. Render Sidebar Navigation
active_page = render_sidebar()

# Page title and contextual description mapping
PAGE_CONFIG = {
    "Overview": (
        "Overview & Intelligence Dashboard",
        "Multi-engine acoustic and neural platform for detecting synthetic voices and authenticating human speech.",
    ),
    "Analyze Audio": (
        "Audio Analysis & Diagnostics",
        "Inspect voice recordings through G.711 telephony codecs and evaluate AASIST, Physics, and Biology signals.",
    ),
    "Engine Insights": (
        "Engine Architecture & Insights",
        "Deep technical exploration of acoustic physics, vocal tract physiology, and spectro-temporal graph attention.",
    ),
    "Dataset Evaluation": (
        "In-The-Wild Research Benchmark",
        "Audited scientific evaluation results across 200 balanced recordings with real ROC/PR curves and limitations.",
    ),
    "Analysis History": (
        "Session Audit History",
        "Historical log of audio analyses performed during your current active session.",
    ),
}

page_title, page_desc = PAGE_CONFIG.get(
    active_page, ("SonicShield", "Acoustic AI Cybersecurity Intelligence")
)

# 6. Render Top Navigation Header Bar
render_top_header(active_page, page_title, page_desc)

# 7. Route and Render Active Page
if active_page == "Overview":
    render_overview_page()
elif active_page == "Analyze Audio":
    render_analyze_page()
elif active_page == "Engine Insights":
    render_insights_page()
elif active_page == "Dataset Evaluation":
    render_evaluation_page()
elif active_page == "Analysis History":
    render_history_page()
else:
    render_overview_page()

# 8. Persistent Ethical & Methodological Footer
footer_html = """
<div class="ss-footer-note">
    <div style="display: flex; align-items: flex-start; gap: 10px;">
        <span style="font-size: 1.1rem; line-height: 1;">🛡️</span>
        <div>
            <strong style="color: #14283D;">Ethical Notice & Prototype Scope:</strong>
            <span style="color: #737B88;">
                SonicShield evaluates acoustic reverberation tails, ambient noise floors, involuntary vocal micro-tremors, 
                and pretrained neural anti-spoofing graph representations. These indicators naturally overlap between genuine human speakers 
                and synthetic audio depending on recording acoustics, microphone quality, and voice compression. 
                Scores represent experimental diagnostic signals to guide human conversational verification, not definitive legal proof of identity or biometric fraud.
            </span>
        </div>
    </div>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
