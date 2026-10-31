"""SonicShield Sidebar Component.

Sophisticated, compact, and visually quiet navigation:
- Constrained width (250px)
- Native Material Icons with high-contrast active state (crisp white on deep navy)
- No bullet points or generic dot characters
- Compact brand mark and system telemetry
"""
from __future__ import annotations

import streamlit as st
from src.pretrained_aasist import get_aasist_classifier

def render_sidebar() -> str:
    """Renders the executive architectural sidebar."""
    aasist = get_aasist_classifier()

    with st.sidebar:
        # Small Geometric Brand Mark
        brand_html = """
        <div style="padding: 0.15rem 0 0.75rem 0; border-bottom: 1px solid #E3E8ED; margin-bottom: 0.75rem;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 26px; height: 26px; border-radius: 6px; background: #14283D; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#13A89E" stroke-width="2" stroke-linejoin="round">
                        <path d="M12 2L4 5V11.09C4 16.14 7.41 20.85 12 22C16.59 20.85 20 16.14 20 11.09V5L12 2Z" fill="#14283D"/>
                        <path d="M8 12V12.01M10 10V14M12 8V16M14 10V14M16 12V12.01" stroke="#FFFFFF" stroke-width="1.8" stroke-linecap="round"/>
                    </svg>
                </div>
                <div>
                    <div style="font-size: 0.98rem; font-weight: 800; letter-spacing: -0.025em; color: #14283D; line-height: 1;">
                        SONICSHIELD
                    </div>
                    <div style="font-size: 0.62rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #13A89E; margin-top: 2px;">
                        Audio Intelligence
                    </div>
                </div>
            </div>
        </div>
        """
        st.markdown(brand_html, unsafe_allow_html=True)

        # Navigation Header
        st.markdown(
            '<div style="font-size: 0.64rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #8E97A4; margin-bottom: 5px; padding-left: 2px;">'
            'NAVIGATION'
            '</div>',
            unsafe_allow_html=True,
        )

        nav_items = [
            ("Overview", "Overview", ":material/dashboard:"),
            ("Analyze Audio", "Analyze Audio", ":material/graphic_eq:"),
            ("Engine Insights", "Engine Insights", ":material/insights:"),
            ("Dataset Evaluation", "Dataset Evaluation", ":material/table_chart:"),
            ("Analysis History", "Analysis History", ":material/history:"),
        ]

        if "current_page" not in st.session_state:
            st.session_state["current_page"] = "Overview"

        for page_id, display_label, icon_name in nav_items:
            is_active = (st.session_state["current_page"] == page_id)
            btn_kind = "primary" if is_active else "secondary"
            if st.button(
                display_label,
                key=f"nav_btn_{page_id}",
                icon=icon_name,
                use_container_width=True,
                type=btn_kind,
            ):
                st.session_state["current_page"] = page_id
                st.rerun()

        # Telemetry Card at Bottom
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size: 0.64rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #8E97A4; margin-bottom: 5px; padding-left: 2px;">'
            'SYSTEM TELEMETRY'
            '</div>',
            unsafe_allow_html=True,
        )

        aasist_state = "Active" if aasist.is_loaded else "Unloaded"
        aasist_color = "#13A89E" if aasist.is_loaded else "#8E97A4"

        telemetry_html = f"""
        <div style="background: #F8FAF9; border: 1px solid #E3E8ED; border-radius: 7px; padding: 8px 10px; font-size: 0.72rem; color: #17212B;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #707C8B;">App Release</span>
                <span style="font-family: 'JetBrains Mono'; font-weight: 600;">v0.3.0</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #707C8B;">AASIST Neural</span>
                <span style="color: {aasist_color}; font-weight: 600;">{aasist_state}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #707C8B;">Rolling Window</span>
                <span style="font-family: 'JetBrains Mono';">6.0s / 1.0s</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #707C8B;">G.711 Telephony</span>
                <span style="color: #13A89E; font-weight: 600;">Enabled</span>
            </div>
        </div>
        """
        st.markdown(telemetry_html, unsafe_allow_html=True)

    return st.session_state["current_page"]
