"""Smoke test for all SonicShield UI page components."""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import streamlit as st
from dashboard.components.overview_page import render_overview_page
from dashboard.components.analyze_page import render_analyze_page
from dashboard.components.insights_page import render_insights_page
from dashboard.components.evaluation_page import render_evaluation_page
from dashboard.components.history_page import render_history_page
from dashboard.components.sidebar import render_sidebar
from dashboard.components.header import render_top_header
from src.pipeline import run_pipeline

print("--- Testing Header & Sidebar ---")
render_sidebar()
render_top_header("Overview", "Test Title", "Test Description")
print("Sidebar & Header OK")

print("--- Testing Overview Page ---")
render_overview_page()
print("Overview Page OK")

print("--- Testing Analyze Page (Empty State) ---")
render_analyze_page()
print("Analyze Page Empty OK")

print("--- Testing Analyze Page (With Real Pipeline Result) ---")
res = run_pipeline("data/smoke/smoke_human_like.wav", codec="g711", detection_mode="pretrained")
st.session_state["active_result"] = res
st.session_state["active_label"] = "smoke_human_like.wav"
render_analyze_page()
print("Analyze Page With Result OK")

print("--- Testing Engine Insights Page ---")
render_insights_page()
print("Engine Insights Page OK")

print("--- Testing Dataset Evaluation Page ---")
render_evaluation_page()
print("Dataset Evaluation Page OK")

print("--- Testing Analysis History Page ---")
st.session_state["analysis_history"] = [{
    "timestamp": "2026-10-31 15:00:00",
    "filename": "smoke_human_like.wav",
    "codec": "G711",
    "mode": "pretrained",
    "score": res["overall_liveness"],
    "band": res["overall_band"],
    "aasist_class": res.get("overall_aasist_class"),
    "aasist_score": res.get("overall_aasist_bona_fide_score"),
    "duration_sec": res["duration_sec"],
    "result_obj": res,
}]
render_history_page()
print("Analysis History Page OK")

print("\nALL 5 UI PAGES AND COMPONENTS PASSED TEST CLEANLY!")
