"""SonicShield Compact Analysis History Page Component.

Maintains session-based telemetry and records for every audio file analyzed
during the current session, allowing instant result recall and inspection.
"""
from __future__ import annotations

import streamlit as st

def render_history_page():
    """Renders the session analysis history table and inspection controls."""
    st.markdown(
        """
        <div class="ss-hero-compact">
            <div class="ss-hero-badge">
                <span>●</span>
                <span>SESSION AUDIT TRAIL</span>
            </div>
            <h1 class="ss-hero-h1">
                Analysis History
            </h1>
            <p class="ss-hero-lead-compact">
                Audio recordings inspected during your current session are logged below. 
                Reopen any previous evaluation to restore its full diagnostic report.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history = st.session_state.get("analysis_history", [])

    if not history:
        with st.container(border=True):
            st.markdown(
                """
                <div style="text-align: center; padding: 2rem 1.5rem;">
                    <h3 style="font-size: 1.05rem; font-weight: 700; color: #14283D; margin-bottom: 4px;">
                        No Analysis Performed in this Session
                    </h3>
                    <p style="font-size: 0.80rem; color: #707C8B; max-width: 440px; margin: 0 auto 16px auto; line-height: 1.45;">
                        Upload a WAV recording or select a preloaded demo benchmark sample on the <em>Analyze Audio</em> page to populate this audit log.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_c1, col_c2, col_c3 = st.columns([1.5, 2, 1.5])
            with col_c2:
                if st.button("Go to Audio Analysis", icon=":material/graphic_eq:", type="primary", use_container_width=True):
                    st.session_state["current_page"] = "Analyze Audio"
                    st.rerun()

    else:
        with st.container(border=True):
            st.markdown(
                f"""
                <div style="margin-bottom: 8px;">
                    <div style="font-size: 0.95rem; font-weight: 700; color: #14283D;">Session Audit Records ({len(history)} files analyzed)</div>
                    <div style="font-size: 0.76rem; color: #707C8B;">Select 'Reopen' on any record to view its full diagnostic report.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            for idx, item in enumerate(history):
                score = item["score"]
                aas_class = item.get("aasist_class")
                cls_color = "#13A89E" if aas_class == "bona_fide" else ("#D95757" if aas_class == "spoof" else "#707C8B")
                cls_text = "Bona-Fide" if aas_class == "bona_fide" else ("Spoof" if aas_class == "spoof" else "N/A")

                col_h1, col_h2, col_h3, col_h4, col_h5 = st.columns([2, 1, 1, 1, 1])

                with col_h1:
                    st.markdown(f"**{item['filename']}**<br><span style='font-size:0.70rem;color:#8E97A4;'>{item['timestamp']}</span>", unsafe_allow_html=True)
                with col_h2:
                    st.markdown(f"Codec: `{item['codec']}`<br><span style='font-size:0.70rem;color:#8E97A4;'>{item['duration_sec']}s</span>", unsafe_allow_html=True)
                with col_h3:
                    st.markdown(f"Neural: <strong style='color:{cls_color};'>{cls_text}</strong>", unsafe_allow_html=True)
                with col_h4:
                    st.markdown(f"Score: **{score} / 100**", unsafe_allow_html=True)
                with col_h5:
                    if st.button("Reopen", key=f"reopen_hist_{idx}", icon=":material/visibility:", use_container_width=True):
                        st.session_state["active_result"] = item["result_obj"]
                        st.session_state["active_label"] = item["filename"]
                        st.session_state["current_page"] = "Analyze Audio"
                        st.rerun()

                st.markdown("<hr style='border:0; border-top:1px solid #EEF1F4; margin:5px 0;'>", unsafe_allow_html=True)

        if st.button("Clear Session History", icon=":material/delete:", type="secondary"):
            st.session_state["analysis_history"] = []
            st.rerun()
