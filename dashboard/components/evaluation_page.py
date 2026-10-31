"""SonicShield Compact Dataset Evaluation Page Component.

Scientific research benchmark dashboard loading real metrics, confusion matrices,
ROC curves, and PR curves from `results/in_the_wild/`.
"""
from __future__ import annotations

import os
import pandas as pd
import streamlit as st

METRICS_CSV = "results/in_the_wild/comparative_metrics_table.csv"
ROC_IMG = "results/in_the_wild/roc_curves_comparison.png"
PR_IMG = "results/in_the_wild/pr_curves_comparison.png"
CONF_IMG = "results/in_the_wild/confusion_matrices_comparison.png"

def render_evaluation_page():
    """Renders the research benchmark page with real empirical evaluation data."""
    st.markdown(
        """
        <div class="ss-hero-compact">
            <div class="ss-hero-badge">
                <span>●</span>
                <span>RESEARCH BENCHMARK & AUDIT</span>
            </div>
            <h1 class="ss-hero-h1">
                In-The-Wild Evaluation Dashboard
            </h1>
            <p class="ss-hero-lead-compact">
                Empirical evaluation of SonicShield's Physics, Biology, AASIST, Baseline, 
                and Experimental Hybrid scoring systems on 200 real-world recordings (Seed 42).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Benchmark Overview Telemetry Grid
    st.markdown(
        """
        <div class="ss-card">
            <div class="ss-card-header">
                <div>
                    <div class="ss-card-title">Dataset Benchmark Specifications</div>
                    <div class="ss-card-subtitle">Audited sample details from the In-The-Wild Audio Deepfake Dataset.</div>
                </div>
                <span class="ss-spec-pill">Seed 42 Reproducible</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px;">
                <div style="background: #F8FAF9; padding: 8px 10px; border-radius: 6px; border: 1px solid #E2E6E9;">
                    <div style="font-size: 0.68rem; color: #5F6876; text-transform: uppercase; font-weight: 600;">Total Evaluated Files</div>
                    <div style="font-size: 1.35rem; font-weight: 800; font-family: 'JetBrains Mono'; color: #14283D;">200</div>
                    <div style="font-size: 0.68rem; color: #16A6A0;">0 Skipped / 0 Corrupted</div>
                </div>
                <div style="background: #F8FAF9; padding: 8px 10px; border-radius: 6px; border: 1px solid #E2E6E9;">
                    <div style="font-size: 0.68rem; color: #5F6876; text-transform: uppercase; font-weight: 600;">Bona-Fide (Genuine)</div>
                    <div style="font-size: 1.35rem; font-weight: 800; font-family: 'JetBrains Mono'; color: #16A6A0;">100</div>
                    <div style="font-size: 0.68rem; color: #5F6876;">50.0% Class Prevalence</div>
                </div>
                <div style="background: #F8FAF9; padding: 8px 10px; border-radius: 6px; border: 1px solid #E2E6E9;">
                    <div style="font-size: 0.68rem; color: #5F6876; text-transform: uppercase; font-weight: 600;">Spoof (Deepfake)</div>
                    <div style="font-size: 1.35rem; font-weight: 800; font-family: 'JetBrains Mono'; color: #D94848;">100</div>
                    <div style="font-size: 0.68rem; color: #5F6876;">50.0% Class Prevalence</div>
                </div>
                <div style="background: #F8FAF9; padding: 8px 10px; border-radius: 6px; border: 1px solid #E2E6E9;">
                    <div style="font-size: 0.68rem; color: #5F6876; text-transform: uppercase; font-weight: 600;">Unique Public Figures</div>
                    <div style="font-size: 1.35rem; font-weight: 800; font-family: 'JetBrains Mono'; color: #14283D;">38</div>
                    <div style="font-size: 0.68rem; color: #D9822B;">21 Cross-Class Overlap</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Performance Comparison Table
    st.markdown(
        """
        <div class="ss-card">
            <div class="ss-card-header">
                <div>
                    <div class="ss-card-title">Empirical Multi-Engine Metrics Comparison</div>
                    <div class="ss-card-subtitle">Exact measured performance across all 5 scoring pipelines on the 200 In-The-Wild sample.</div>
                </div>
                <span class="ss-spec-pill">Threshold = 0.5</span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    if os.path.exists(METRICS_CSV):
        df_metrics = pd.read_csv(METRICS_CSV)
        formatted_df = df_metrics[[
            "Engine / System", "Accuracy", "ROC-AUC", "Bona-fide Precision", "Bona-fide Recall",
            "Macro F1", "TP (Live)", "TN (Spoof)", "FP (Fake as Live)", "FN (Live as Fake)", "EER"
        ]].copy()

        formatted_df["Accuracy"] = formatted_df["Accuracy"].apply(lambda v: f"{v*100:.1f}%")
        formatted_df["ROC-AUC"] = formatted_df["ROC-AUC"].apply(lambda v: f"{v:.4f}")
        formatted_df["Bona-fide Precision"] = formatted_df["Bona-fide Precision"].apply(lambda v: f"{v*100:.1f}%")
        formatted_df["Bona-fide Recall"] = formatted_df["Bona-fide Recall"].apply(lambda v: f"{v*100:.1f}%")
        formatted_df["Macro F1"] = formatted_df["Macro F1"].apply(lambda v: f"{v:.4f}")
        formatted_df["EER"] = formatted_df["EER"].apply(lambda v: f"{v*100:.1f}%")

        st.dataframe(formatted_df, use_container_width=True, hide_index=True)
    else:
        st.warning(f"Metrics table not found at `{METRICS_CSV}`.")

    st.markdown("</div>", unsafe_allow_html=True)

    # 3. Visualizations Tabs
    st.markdown(
        """
        <div class="ss-card">
            <div class="ss-card-header">
                <div>
                    <div class="ss-card-title">Diagnostic Evaluation Visualizations</div>
                    <div class="ss-card-subtitle">Receiver Operating Characteristic, Precision-Recall, and Confusion Matrices.</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    tab_roc, tab_pr, tab_conf = st.tabs(["📉 ROC Curves (AUC)", "🎯 Precision-Recall Curves", "🔲 Confusion Matrices"])

    with tab_roc:
        if os.path.exists(ROC_IMG):
            st.image(ROC_IMG, caption="Multi-Engine ROC Curves (AASIST leads with AUC = 0.6225 vs Hybrid = 0.5712)", use_container_width=True)
        else:
            st.info("ROC Curve image not found.")

    with tab_pr:
        if os.path.exists(PR_IMG):
            st.image(PR_IMG, caption="Precision-Recall Curves Comparison (AASIST AP = 0.6677 vs Hybrid = 0.5624)", use_container_width=True)
        else:
            st.info("Precision-Recall Curve image not found.")

    with tab_conf:
        if os.path.exists(CONF_IMG):
            st.image(CONF_IMG, caption="Four-Engine Confusion Matrices Quad-Plot at Threshold 0.5 (AASIST and Hybrid both reject 36 fakes)", use_container_width=True)
        else:
            st.info("Confusion matrix image not found.")

    st.markdown("</div>", unsafe_allow_html=True)

    # 4. Threshold Sweep Table
    st.markdown(
        """
        <div class="ss-card">
            <div class="ss-card-header">
                <div>
                    <div class="ss-card-title">Threshold Sweep Progression (AASIST Engine)</div>
                    <div class="ss-card-subtitle">How False Acceptance Rate (FAR) and False Rejection Rate (FRR) shift across thresholds [0.1 to 0.9].</div>
                </div>
            </div>
            <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; font-size: 0.78rem; text-align: center;">
                    <thead>
                        <tr style="border-bottom: 2px solid #E2E6E9; color: #14283D; font-weight: 700;">
                            <th style="padding: 6px;">Threshold</th>
                            <th style="padding: 6px;">FAR (Spoof Accepted)</th>
                            <th style="padding: 6px;">FRR (Genuine Rejected)</th>
                            <th style="padding: 6px;">Accuracy</th>
                            <th style="padding: 6px;">Precision</th>
                            <th style="padding: 6px;">Recall</th>
                            <th style="padding: 6px;">F1-Score</th>
                            <th style="padding: 6px;">TP</th>
                            <th style="padding: 6px;">FP</th>
                            <th style="padding: 6px;">TN</th>
                            <th style="padding: 6px;">FN</th>
                        </tr>
                    </thead>
                    <tbody style="font-family: 'JetBrains Mono', monospace; color: #151A22;">
                        <tr style="border-bottom: 1px solid #EEF1F2;"><td>0.1</td><td>73.0%</td><td>15.0%</td><td>56.0%</td><td>53.8%</td><td>85.0%</td><td>0.6589</td><td>85</td><td>73</td><td>27</td><td>15</td></tr>
                        <tr style="border-bottom: 1px solid #EEF1F2;"><td>0.2</td><td>70.0%</td><td>16.0%</td><td>57.0%</td><td>54.6%</td><td>84.0%</td><td>0.6614</td><td>84</td><td>70</td><td>30</td><td>16</td></tr>
                        <tr style="border-bottom: 1px solid #EEF1F2;"><td>0.3</td><td>68.0%</td><td>17.0%</td><td>57.5%</td><td>55.0%</td><td>83.0%</td><td>0.6614</td><td>83</td><td>68</td><td>32</td><td>17</td></tr>
                        <tr style="border-bottom: 1px solid #EEF1F2; background: #F8FAF9;"><td>0.4</td><td>65.0%</td><td>17.0%</td><td>59.0%</td><td>56.1%</td><td>83.0%</td><td>0.6694</td><td>83</td><td>65</td><td>35</td><td>17</td></tr>
                        <tr style="border-bottom: 1px solid #EEF1F2; background: rgba(22, 166, 160, 0.08); font-weight: 700;"><td>0.5 (Default)</td><td>64.0%</td><td>23.0%</td><td>56.5%</td><td>54.6%</td><td>77.0%</td><td>0.6390</td><td>77</td><td>64</td><td>36</td><td>23</td></tr>
                        <tr style="border-bottom: 1px solid #EEF1F2;"><td>0.6</td><td>64.0%</td><td>24.0%</td><td>56.0%</td><td>54.3%</td><td>76.0%</td><td>0.6333</td><td>76</td><td>64</td><td>36</td><td>24</td></tr>
                        <tr style="border-bottom: 1px solid #EEF1F2;"><td>0.7</td><td>60.0%</td><td>27.0%</td><td>56.5%</td><td>54.9%</td><td>73.0%</td><td>0.6266</td><td>73</td><td>60</td><td>40</td><td>27</td></tr>
                        <tr style="border-bottom: 1px solid #EEF1F2;"><td>0.8</td><td>57.0%</td><td>32.0%</td><td>55.5%</td><td>54.4%</td><td>68.0%</td><td>0.6044</td><td>68</td><td>57</td><td>43</td><td>32</td></tr>
                        <tr><td>0.9</td><td>49.0%</td><td>37.0%</td><td>57.0%</td><td>56.3%</td><td>63.0%</td><td>0.5943</td><td>63</td><td>49</td><td>51</td><td>37</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 5. Scientific Limitations (7 Clear Points)
    st.markdown(
        """
        <div class="ss-card" style="border-left: 3px solid #D9822B;">
            <div class="ss-card-header">
                <div>
                    <div class="ss-card-title">⚠️ Scientific Limitations & Evaluation Integrity</div>
                    <div class="ss-card-subtitle">Transparent disclosure of benchmark constraints and technical audit findings.</div>
                </div>
                <span class="ss-status-pill warning">Audit Findings</span>
            </div>
            <div style="font-size: 0.80rem; color: #151A22; line-height: 1.55;">
                <ol style="padding-left: 18px; margin: 0;">
                    <li><strong>Sample Size:</strong> The benchmark evaluated a balanced sample of 200 recordings (100 genuine, 100 deepfake), not the full 31,779 recordings of the In-The-Wild dataset.</li>
                    <li><strong>Speaker Overlap:</strong> 21 public figures appear in both classes (accounting for 171 of 200 recordings). On the disjoint 29 recordings, performance drops to 44.8%, showing that models partially memorize speaker timbral cues.</li>
                    <li><strong>AASIST Baseline:</strong> Pretrained AASIST achieved <strong>56.5% accuracy</strong>, <strong>ROC-AUC 0.6225</strong>, and <strong>EER 44.0%</strong> on this in-the-wild sample.</li>
                    <li><strong>Hybrid Distortion:</strong> The experimental hybrid achieved 57.5% accuracy (+1.0%), but its <strong>ROC-AUC dropped to 0.5712</strong> and EER worsened to 46.5%, demonstrating that naive linear averaging dilutes AASIST's discriminative power.</li>
                    <li><strong>Identical Spoof Rejection:</strong> At the default 0.5 threshold, both AASIST and the Hybrid catch the <strong>exact same number of fake recordings (36 out of 100)</strong>. The hybrid provides zero additional protection against deepfakes.</li>
                    <li><strong>Uncalibrated Scores:</strong> None of the engine outputs are calibrated probabilities. Deep neural softmax outputs are overconfident under out-of-domain shift.</li>
                    <li><strong>Prototype Status:</strong> The current benchmark results demonstrate that SonicShield is an exploratory prototype, not a production biometric security guarantee.</li>
                </ol>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
