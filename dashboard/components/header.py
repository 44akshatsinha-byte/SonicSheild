"""SonicShield Compact Top Header Component.

Provides a unified single-line architectural header:
- Small breadcrumb + page title
- Inline status pills for FastAPI and AASIST (no line wrapping / oval deformation)
- Sized cleanly to fit comfortably without vertical bloat
"""
from __future__ import annotations

import requests
import streamlit as st
from src.pretrained_aasist import get_aasist_classifier

def get_backend_status() -> dict:
    """Checks the FastAPI backend status via HTTP or local health check."""
    try:
        resp = requests.get("http://127.0.0.1:8000/health", timeout=0.20)
        if resp.status_code == 200:
            return {
                "online": True,
                "label": "API Live (:8000)",
            }
    except Exception:
        pass

    try:
        from api.main import health_check
        health_check()
        return {
            "online": True,
            "label": "API Live (Local)",
        }
    except Exception:
        return {
            "online": False,
            "label": "API Offline",
        }

def render_top_header(current_page: str, page_title: str, page_description: str):
    """Renders the sleek, compact top navigation header bar."""
    backend_info = get_backend_status()
    aasist = get_aasist_classifier()

    api_class = "active" if backend_info["online"] else "inactive"
    api_dot = "●" if backend_info["online"] else "○"
    api_label = backend_info["label"]

    aasist_loaded = aasist.is_loaded
    aasist_class = "active" if aasist_loaded else "inactive"
    aasist_dot = "●" if aasist_loaded else "○"
    aasist_label = "AASIST Active" if aasist_loaded else "AASIST Unloaded"

    header_html = f"""
    <div class="ss-top-header">
        <div class="ss-header-left">
            <div class="ss-breadcrumb">
                <span>SONICSHIELD</span>
                <span>/</span>
                <span style="color: var(--ss-navy-deep);">{current_page.upper()}</span>
            </div>
            <span style="color: var(--ss-border); font-size: 0.8rem;">|</span>
            <div class="ss-header-title">{page_title}</div>
            <span class="ss-header-desc">— {page_description}</span>
        </div>
        <div class="ss-status-group">
            <span class="ss-status-pill {api_class}" title="FastAPI Health Telemetry">
                <span style="font-size: 8px;">{api_dot}</span>
                <span>{api_label}</span>
            </span>
            <span class="ss-status-pill {aasist_class}" title="AASIST Neural Checkpoint Status">
                <span style="font-size: 8px;">{aasist_dot}</span>
                <span>{aasist_label}</span>
            </span>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
