"""SonicShield Compact Engine Insights Page Component.

"Three Perspectives. One Audio Signal."
Deep-dive into Physics, Biology, and AASIST Neural Engines with an architectural
data-flow diagram and scientific explanations.
"""
from __future__ import annotations

import streamlit as st

def render_insights_page():
    """Renders the balanced, compact technology deep-dive and architecture overview."""
    st.markdown(
        """<div class="ss-hero-compact">
<div class="ss-hero-badge">
<span>●</span>
<span>SYSTEM ARCHITECTURE & DEEP DIVE</span>
</div>
<h1 class="ss-hero-h1">
Three Perspectives. One Audio Signal.
</h1>
<p class="ss-hero-lead-compact">
SonicShield analyzes incoming audio across acoustic physics, human vocal physiology, 
and deep neural representations. Rather than relying on a single black-box score, 
we provide layered, decoupled diagnostic telemetry.
</p>
</div>""",
        unsafe_allow_html=True,
    )

    # 1. System Architecture Diagram (Clean Unindented SVG)
    svg_html = """<div class="ss-card">
<div class="ss-card-header">
<div>
<div class="ss-card-title">End-to-End Signal Pipeline Architecture</div>
<div class="ss-card-subtitle">Transparent signal flow from raw telephony ingress to decoupled diagnostic outputs.</div>
</div>
<span class="ss-spec-pill">Pipeline Flow</span>
</div>
<div style="text-align: center; padding: 0.6rem 0; overflow-x: auto;">
<svg width="820" height="180" viewBox="0 0 820 180" fill="none" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%;">
<!-- Ingress Block -->
<rect x="10" y="55" width="120" height="65" rx="7" fill="#FFFFFF" stroke="#14283D" stroke-width="1.8"/>
<text x="70" y="82" font-family="'Plus Jakarta Sans', sans-serif" font-weight="700" font-size="11" fill="#14283D" text-anchor="middle">Audio Ingress</text>
<text x="70" y="98" font-family="'JetBrains Mono', monospace" font-size="9" fill="#707C8B" text-anchor="middle">WAV / Stream</text>
<text x="70" y="110" font-family="'JetBrains Mono', monospace" font-size="8.5" fill="#13A89E" text-anchor="middle">Any Sample Rate</text>

<!-- Arrow 1 -->
<path d="M130 87 H 165" stroke="#CBD5E1" stroke-width="1.8"/>

<!-- Preprocessing Block -->
<rect x="170" y="55" width="135" height="65" rx="7" fill="#FFFFFF" stroke="#E3E8ED" stroke-width="1.8"/>
<text x="237" y="80" font-family="'Plus Jakarta Sans', sans-serif" font-weight="700" font-size="11" fill="#14283D" text-anchor="middle">Audio Preprocessing</text>
<text x="237" y="96" font-family="'JetBrains Mono', monospace" font-size="9" fill="#707C8B" text-anchor="middle">Stereo to Mono</text>
<text x="237" y="109" font-family="'JetBrains Mono', monospace" font-size="8.5" fill="#707C8B" text-anchor="middle">Peak Normalization</text>

<!-- Arrow 2 (Splits into 3) -->
<path d="M305 87 H 330 V 28 H 360" stroke="#CBD5E1" stroke-width="1.6"/>
<path d="M305 87 H 360" stroke="#CBD5E1" stroke-width="1.6"/>
<path d="M305 87 H 330 V 147 H 360" stroke="#CBD5E1" stroke-width="1.6"/>

<!-- Engine 1: Physics -->
<rect x="365" y="6" width="180" height="46" rx="6" fill="#FFFFFF" stroke="#0284C7" stroke-width="1.8"/>
<text x="455" y="26" font-family="'Plus Jakarta Sans', sans-serif" font-weight="700" font-size="11" fill="#0284C7" text-anchor="middle">1. Physics Engine</text>
<text x="455" y="40" font-family="'JetBrains Mono', monospace" font-size="8.5" fill="#707C8B" text-anchor="middle">Room Acoustics & Reverberation</text>

<!-- Engine 2: AASIST (Primary) -->
<rect x="365" y="65" width="180" height="46" rx="6" fill="#FFFFFF" stroke="#13A89E" stroke-width="2.2"/>
<text x="455" y="85" font-family="'Plus Jakarta Sans', sans-serif" font-weight="800" font-size="11" fill="#13A89E" text-anchor="middle">2. AASIST Neural Engine ★</text>
<text x="455" y="99" font-family="'JetBrains Mono', monospace" font-size="8.5" fill="#14283D" text-anchor="middle">Graph Attention • 16 kHz Waveform</text>

<!-- Engine 3: Biology -->
<rect x="365" y="125" width="180" height="46" rx="6" fill="#FFFFFF" stroke="#8B5CF6" stroke-width="1.8"/>
<text x="455" y="145" font-family="'Plus Jakarta Sans', sans-serif" font-weight="700" font-size="11" fill="#8B5CF6" text-anchor="middle">3. Biology Engine</text>
<text x="455" y="159" font-family="'JetBrains Mono', monospace" font-size="8.5" fill="#707C8B" text-anchor="middle">Vocal Tract Jitter & Shimmer</text>

<!-- Connectors to Output -->
<path d="M545 28 H 575 V 87 H 600" stroke="#CBD5E1" stroke-width="1.6"/>
<path d="M545 87 H 600" stroke="#13A89E" stroke-width="2"/>
<path d="M545 147 H 575 V 87 H 600" stroke="#CBD5E1" stroke-width="1.6"/>

<!-- Output Decision Block -->
<rect x="605" y="52" width="200" height="72" rx="8" fill="#F8FAF9" stroke="#14283D" stroke-width="1.8"/>
<text x="705" y="76" font-family="'Plus Jakarta Sans', sans-serif" font-weight="800" font-size="11" fill="#14283D" text-anchor="middle">Detection Interpretation</text>
<text x="705" y="92" font-family="'JetBrains Mono', monospace" font-size="8.5" fill="#13A89E" text-anchor="middle">Primary: AASIST Neural Score</text>
<text x="705" y="105" font-family="'JetBrains Mono', monospace" font-size="8" fill="#707C8B" text-anchor="middle">Diagnostics: Physics & Biology</text>
<text x="705" y="117" font-family="'JetBrains Mono', monospace" font-size="8" fill="#D99A35" text-anchor="middle">Decoupled Uncalibrated Signals</text>
</svg>
</div>
<div style="font-size: 0.74rem; color: #707C8B; border-top: 1px solid #EEF1F4; padding-top: 8px;">
💡 <strong>Architectural Note:</strong> AASIST acts as the primary neural deepfake discriminator. Physics and Biology engines serve as supporting acoustic diagnostics to explain signal characteristics without forcing an uncalibrated weighted average.
</div>
</div>"""
    st.markdown(svg_html, unsafe_allow_html=True)

    # 2. Detailed Engine Cards (Physics & Biology)
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown(
            """<div class="ss-card" style="border-top: 3px solid #0284C7; height: 100%;">
<div class="ss-card-header">
<div>
<span class="ss-engine-tag physics">Acoustic Physics</span>
<div class="ss-card-title">Physics Engine: Acoustic Environment</div>
</div>
</div>
<p style="font-size: 0.78rem; color: #17212B; line-height: 1.45;">
Every physical human voice reverberates within a physical room. As sound waves strike boundaries, 
they produce an <strong>Energy Decay Curve (EDC)</strong> intrinsically linked to the physical environment.
</p>
<div style="margin: 8px 0;">
<div style="font-size: 0.70rem; font-weight: 700; color: #0284C7; margin-bottom: 2px;">KEY ACOUSTIC SIGNALS:</div>
<ul style="font-size: 0.75rem; color: #707C8B; padding-left: 18px; line-height: 1.5; margin: 0;">
<li><strong>Early Reflection Tails:</strong> Micro-reverberation (0–50 ms) after phonemes stop.</li>
<li><strong>Ambient Room Tone:</strong> Noise floor during natural conversational pauses.</li>
<li><strong>Spectral Rolloff & Flatness:</strong> Identifies unnatural cutoff or synthetic silence.</li>
</ul>
</div>
<div style="background: #F0F9FF; border: 1px solid #BAE6FD; border-radius: 6px; padding: 7px 10px; font-size: 0.70rem; color: #0369A1;">
<strong>Context:</strong> Synthetic audio rendered directly from digital vocoders often lacks natural room acoustics (producing dry speech).
</div>
</div>""",
            unsafe_allow_html=True,
        )

    with col_p2:
        st.markdown(
            """<div class="ss-card" style="border-top: 3px solid #8B5CF6; height: 100%;">
<div class="ss-card-header">
<div>
<span class="ss-engine-tag biology">Vocal Physiology</span>
<div class="ss-card-title">Biology Engine: Vocal Biomarkers</div>
</div>
</div>
<p style="font-size: 0.78rem; color: #17212B; line-height: 1.45;">
Human vocal cords cannot sustain mathematically perfect vibration. 
Involuntary muscular tremors, mucosal wave turbulence, and pulmonary cycles create micro-perturbations.
</p>
<div style="margin: 8px 0;">
<div style="font-size: 0.70rem; font-weight: 700; color: #8B5CF6; margin-bottom: 2px;">KEY BIOLOGICAL SIGNALS:</div>
<ul style="font-size: 0.75rem; color: #707C8B; padding-left: 18px; line-height: 1.5; margin: 0;">
<li><strong>Pitch Perturbation (Jitter):</strong> Cycle-to-cycle frequency variations in human vocal cords.</li>
<li><strong>Amplitude Perturbation (Shimmer):</strong> Peak amplitude irregularities in voiced phonemes.</li>
<li><strong>Harmonic-to-Noise Ratio (HNR):</strong> Ratio between glottal energy and turbulent aspiration.</li>
</ul>
</div>
<div style="background: #FAF5FF; border: 1px solid #E9D5FF; border-radius: 6px; padding: 7px 10px; font-size: 0.70rem; color: #6D28D9;">
<strong>Context:</strong> Advanced vocoders (HiFi-GAN, VITS) now synthesize realistic pitch variation, making this a supporting diagnostic.
</div>
</div>""",
            unsafe_allow_html=True,
        )

    # 3. AASIST Neural Engine Deep Dive
    st.markdown(
        """<div class="ss-card" style="border-top: 3px solid var(--ss-teal-accent);">
<div class="ss-card-header">
<div>
<span class="ss-engine-tag aasist">Primary Neural Anti-Spoofing</span>
<div class="ss-card-title">Pretrained AASIST: Integrated Spectro-Temporal Graph Attention</div>
<div class="ss-card-subtitle">Jung et al., Clova AI / ICASSP 2022 • Checkpoint: models/pretrained/AASIST.pth</div>
</div>
<span class="ss-spec-pill">Deep Neural Model</span>
</div>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px;">
<div>
<p style="font-size: 0.78rem; color: #17212B; line-height: 1.45;">
AASIST operates directly on raw audio waveforms using a convolutional front-end with SincNet filters 
to extract spectral features across the audible bandwidth, followed by a bidirectional graph attention network (GAT).
</p>
<div style="font-size: 0.74rem; color: #707C8B; line-height: 1.5;">
• <strong>Spectral Graph:</strong> Analyzes inter-frequency dependencies to spot artificial phase discontinuity.
<br>
• <strong>Temporal Graph:</strong> Examines time-domain continuity across speech segments.
<br>
• <strong>Graph Pooling:</strong> Integrates spectro-temporal nodes into final 2-dimensional class logits:
<br>
<code style="display:inline-block; margin-top:2px; padding:2px 6px; background:#F4F6F7; border-radius:4px; font-size:0.72rem;">Logits: [z_0 (Spoof), z_1 (Bona-Fide)]</code>
</div>
</div>
<div>
<div style="background: #F8FAF9; border: 1px solid #E3E8ED; border-radius: 7px; padding: 9px 11px;">
<div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: #14283D; margin-bottom: 3px;">
Telephony Ingestion Pipeline
</div>
<div style="font-size: 0.73rem; color: #707C8B; line-height: 1.42;">
1. Incoming audio is processed through simulated telephony codec (G.711 μ-law @ 8 kHz).
<br>
2. Codec-processed audio is resampled to <strong>16,000 Hz</strong> using polyphase filtering.
<br>
3. Formats exactly <strong>64,600 samples (~4.04s)</strong> for deterministic inference.
<br>
4. Softmax produces bona-fide evidence score:
$$\\sigma(z_1) = \\frac{{e^{{z_1}}}}{{e^{{z_0}} + e^{{z_1}}}}$$
</div>
</div>
</div>
</div>
<div style="margin-top: 8px; padding-top: 7px; border-top: 1px solid #EEF1F4; font-size: 0.72rem; color: #707C8B;">
⚠️ <strong>Telephony Note:</strong> Resampling 8 kHz telephone audio to 16 kHz does not restore high frequencies discarded by the PSTN network. The model operates under out-of-domain conditions.
</div>
</div>""",
        unsafe_allow_html=True,
    )
