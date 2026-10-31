"""SonicShield Audio Charts and Visualizations.

Tailored for high visual depth with selective dark console styling for audio waveforms,
rolling window timelines, and compact Merkur-inspired gauges.
"""
from __future__ import annotations

import io
import soundfile as sf
import numpy as np
import plotly.graph_objects as go

def audio_to_wav_bytes(audio: np.ndarray, sr: int = 8000) -> bytes:
    """Converts a numpy audio array to 16-bit PCM WAV bytes for st.audio playback."""
    buf = io.BytesIO()
    clipped = np.clip(audio, -1.0, 1.0).astype(np.float32)
    sf.write(buf, clipped, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()

def render_merkur_gauge(score: int, band: str = "green", title: str = "", subtitle: str = "") -> go.Figure:
    """Creates a compact, architectural gauge indicator."""
    if band == "green":
        bar_color = "#16A6A0"      # Teal
    elif band == "amber":
        bar_color = "#D9822B"      # Amber
    else:
        bar_color = "#D94848"      # Danger Coral

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": f"<b>{title}</b><br><span style='font-size:10px;color:#5F6876'>{subtitle}</span>", "font": {"size": 12, "color": "#14283D", "family": "Plus Jakarta Sans"}},
            domain={"x": [0, 1], "y": [0, 1]},
            number={"suffix": "", "font": {"size": 30, "color": "#14283D", "family": "Plus Jakarta Sans"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#CBD5E1", "ticks": "outside", "tickfont": {"size": 8, "color": "#5F6876"}},
                "bar": {"color": bar_color, "thickness": 0.22},
                "bgcolor": "#F4F6F7",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(217, 72, 72, 0.08)"},
                    {"range": [40, 70], "color": "rgba(217, 130, 43, 0.08)"},
                    {"range": [70, 100], "color": "rgba(22, 166, 160, 0.08)"},
                ],
                "threshold": {
                    "line": {"color": bar_color, "width": 2.5},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=150,
        margin={"l": 15, "r": 15, "t": 28, "b": 8},
    )
    return fig

def render_timeline_chart(windows: list[dict], show_ml: bool = False, show_aasist: bool = True) -> go.Figure:
    """Builds a crisp timeline chart showing score progression across rolling 6s windows."""
    t_vals = [w["t"] for w in windows]
    baseline_vals = [w["baseline_score"] for w in windows]
    physics_vals = [w["physics"] * 100.0 for w in windows]
    biology_vals = [w["biology"] * 100.0 for w in windows]
    aasist_vals = [w.get("aasist_score") for w in windows]
    ml_vals = [w.get("ml_score") for w in windows]

    fig = go.Figure()

    # Band background regions (subtle)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(22, 166, 160, 0.05)", line_width=0, layer="below")
    fig.add_hrect(y0=40, y1=70, fillcolor="rgba(217, 130, 43, 0.05)", line_width=0, layer="below")
    fig.add_hrect(y0=0, y1=40, fillcolor="rgba(217, 72, 72, 0.05)", line_width=0, layer="below")

    # Threshold dashed guidelines
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(22, 166, 160, 0.35)", line_width=1)
    fig.add_hline(y=40, line_dash="dash", line_color="rgba(217, 130, 43, 0.35)", line_width=1)

    # Sub-Engine lines
    fig.add_trace(
        go.Scatter(
            x=t_vals,
            y=physics_vals,
            mode="lines",
            name="Physics (Room)",
            line={"color": "#0284C7", "width": 1.5, "dash": "dot"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_vals,
            y=biology_vals,
            mode="lines",
            name="Biology (Vocal)",
            line={"color": "#9333EA", "width": 1.5, "dash": "dot"},
        )
    )

    # Rule-Based Baseline
    fig.add_trace(
        go.Scatter(
            x=t_vals,
            y=baseline_vals,
            mode="lines+markers",
            name="Baseline",
            line={"color": "#14283D", "width": 2.0},
            marker={"size": 4, "color": "#14283D"},
        )
    )

    # AASIST Neural Engine
    if show_aasist and any(v is not None for v in aasist_vals):
        fig.add_trace(
            go.Scatter(
                x=t_vals,
                y=[v if v is not None else np.nan for v in aasist_vals],
                mode="lines+markers",
                name="AASIST Neural",
                line={"color": "#16A6A0", "width": 2.4},
                marker={"size": 5, "color": "#16A6A0", "symbol": "diamond"},
            )
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        xaxis={
            "title": "Timeline (seconds)",
            "gridcolor": "#EEF1F2",
            "zeroline": False,
            "color": "#5F6876",
            "title_font": {"size": 10, "family": "Plus Jakarta Sans"},
            "tickfont": {"size": 9, "family": "JetBrains Mono"},
        },
        yaxis={
            "title": "Score Index (0 - 100)",
            "range": [-2, 105],
            "gridcolor": "#EEF1F2",
            "zeroline": False,
            "color": "#5F6876",
            "title_font": {"size": 10, "family": "Plus Jakarta Sans"},
            "tickfont": {"size": 9, "family": "JetBrains Mono"},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"color": "#151A22", "size": 10, "family": "Plus Jakarta Sans"},
        },
        height=220,
        margin={"l": 25, "r": 15, "t": 25, "b": 28},
    )
    return fig

def render_dark_waveform(audio: np.ndarray, sr: int = 8000, title: str = "TELEPHONY AUDIO WAVEFORM (8 kHz G.711)") -> go.Figure:
    """Builds a high-precision dark console waveform visualization for visual depth."""
    step = max(1, len(audio) // 1000)
    downsampled = audio[::step]
    t = np.arange(len(downsampled)) * (step / sr)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=downsampled,
            mode="lines",
            name="Audio Signal",
            line={"color": "#13A89E", "width": 1.4},
            fill="tozeroy",
            fillcolor="rgba(19, 168, 158, 0.15)",
        )
    )

    fig.update_layout(
        title={"text": f"<span style='font-family:JetBrains Mono;font-size:10.5px;font-weight:600;color:#CBD5E1;'>{title}</span>", "x": 0.02, "y": 0.95},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0A101A",
        xaxis={
            "title": "Time (s)",
            "gridcolor": "rgba(255, 255, 255, 0.06)",
            "zeroline": True,
            "zerolinecolor": "rgba(255, 255, 255, 0.15)",
            "color": "#94A3B8",
            "tickfont": {"size": 9, "family": "JetBrains Mono"},
            "title_font": {"size": 9, "color": "#94A3B8"},
        },
        yaxis={
            "title": "Amplitude",
            "range": [-1.05, 1.05],
            "gridcolor": "rgba(255, 255, 255, 0.06)",
            "color": "#94A3B8",
            "tickfont": {"size": 8, "family": "JetBrains Mono"},
            "title_font": {"size": 9, "color": "#94A3B8"},
        },
        height=150,
        margin={"l": 25, "r": 15, "t": 25, "b": 25},
        showlegend=False,
    )
    return fig
