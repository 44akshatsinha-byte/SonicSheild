"""SonicShield Design System & CSS Stylesheet.

Engineered to executive cybersecurity SaaS standards inspired by Merkur Design:
- Strict 8px grid & cohesive palette: #F4F6F7, #FFFFFF, #14283D, #17212B, #707C8B, #13A89E, #E7F5F3, #E3E8ED
- Proportioned sidebar (250px) with Material Icons and crisp active contrast
- Light native file uploader & high-contrast selectbox popovers (eliminating dark clashes)
- Structured card system via native Streamlit containers and custom panels
- Selective dark consoles for audio waveforms and technical telemetry
- Tuned for 1366x768 laptop and 1080p desktop displays without excessive scrolling
"""

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --ss-bg-main: #F4F6F7;
        --ss-bg-surface: #FFFFFF;
        --ss-bg-dark: #0E1724;
        --ss-navy-deep: #14283D;
        --ss-text-primary: #17212B;
        --ss-text-secondary: #707C8B;
        --ss-text-muted: #8E97A4;
        --ss-border: #E3E8ED;
        --ss-border-subtle: #EEF1F4;
        --ss-teal-accent: #13A89E;
        --ss-teal-soft: #E7F5F3;
        --ss-amber-warning: #D99A35;
        --ss-amber-soft: #FDF6EC;
        --ss-danger-coral: #D95757;
        --ss-danger-soft: #FDF2F2;
        --ss-shadow-sm: 0 1px 2px rgba(20, 40, 61, 0.04), 0 2px 6px rgba(20, 40, 61, 0.02);
        --ss-shadow: 0 2px 4px rgba(20, 40, 61, 0.04), 0 6px 16px rgba(20, 40, 61, 0.03);
    }

    /* 1. Global Reset & Canvas */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        background-color: var(--ss-bg-main) !important;
        color: var(--ss-text-primary) !important;
        -webkit-font-smoothing: antialiased;
    }

    .stApp {
        background-color: var(--ss-bg-main) !important;
    }

    /* Transparent native header bar (removes Streamlit default black line) */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 1.6rem !important;
        z-index: 10 !important;
    }

    /* Main Container Padding — Optimized for 1366x768 & 1080p */
    .block-container {
        padding-top: 0.6rem !important;
        padding-bottom: 2.2rem !important;
        padding-left: 1.8rem !important;
        padding-right: 1.8rem !important;
        max-width: 1240px !important;
    }

    /* 2. Constrained Sidebar (250px) */
    section[data-testid="stSidebar"] {
        width: 250px !important;
        min-width: 250px !important;
        max-width: 250px !important;
        background: #FFFFFF !important;
        border-right: 1px solid var(--ss-border) !important;
    }

    section[data-testid="stSidebar"] > div {
        width: 250px !important;
        padding: 0.85rem 0.75rem 1.15rem 0.75rem !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] {
        display: none !important;
    }

    /* 3. Typography Hierarchy */
    h1 {
        font-size: 1.55rem !important;
        font-weight: 800 !important;
        color: var(--ss-navy-deep) !important;
        letter-spacing: -0.03em !important;
        margin-bottom: 0.3rem !important;
        line-height: 1.2 !important;
    }

    h2 {
        font-size: 1.25rem !important;
        font-weight: 750 !important;
        color: var(--ss-navy-deep) !important;
        letter-spacing: -0.025em !important;
        margin-bottom: 0.25rem !important;
        line-height: 1.25 !important;
    }

    h3 {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: var(--ss-navy-deep) !important;
        letter-spacing: -0.02em !important;
        margin-bottom: 0.2rem !important;
    }

    h4 {
        font-size: 0.92rem !important;
        font-weight: 650 !important;
        color: var(--ss-navy-deep) !important;
        margin-bottom: 0.15rem !important;
    }

    p, span, label, div {
        color: var(--ss-text-primary);
        font-size: 0.86rem;
    }

    code, kbd, samp, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* 4. Streamlit Buttons & High-Contrast Navigation */
    .stButton > button {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        padding: 0.40rem 0.85rem !important;
        transition: all 0.12s ease !important;
        box-shadow: none !important;
        margin: 1px 0 !important;
        display: flex !important;
        align-items: center !important;
        text-align: left !important;
    }

    /* Active / Primary Button: Solid Deep Navy + Crisp Pure White Text & Icon */
    .stButton > button[kind="primary"] {
        background: #14283D !important;
        border: 1px solid #14283D !important;
        box-shadow: 0 1px 3px rgba(20, 40, 61, 0.18) !important;
    }

    .stButton > button[kind="primary"] *,
    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] span,
    .stButton > button[kind="primary"] div,
    .stButton > button[kind="primary"] [data-testid="stIconMaterial"] {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        fill: #FFFFFF !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: #1C354E !important;
        border-color: #1C354E !important;
    }

    .stButton > button[kind="primary"]:hover * {
        color: #FFFFFF !important;
    }

    /* Inactive / Secondary Button: Clean quiet item */
    .stButton > button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid transparent !important;
    }

    .stButton > button[kind="secondary"] *,
    .stButton > button[kind="secondary"] p,
    .stButton > button[kind="secondary"] span,
    .stButton > button[kind="secondary"] div,
    .stButton > button[kind="secondary"] [data-testid="stIconMaterial"] {
        color: #555E6D !important;
        font-weight: 500 !important;
        fill: #555E6D !important;
    }

    .stButton > button[kind="secondary"]:hover {
        background: #F1F4F5 !important;
        border-color: #E3E8ED !important;
    }

    .stButton > button[kind="secondary"]:hover * {
        color: #14283D !important;
        font-weight: 600 !important;
    }

    /* 5. Streamlit Native Bordered Containers — Unified Card Surface */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background: #FFFFFF !important;
        border: 1px solid var(--ss-border) !important;
        border-radius: 9px !important;
        padding: 1rem 1.15rem !important;
        box-shadow: var(--ss-shadow-sm) !important;
        margin-bottom: 0.85rem !important;
    }

    /* 6. File Uploader Styling (Eliminates Dark Background Clashes) */
    div[data-testid="stFileUploader"] {
        background: transparent !important;
        padding: 0 !important;
        border: none !important;
    }

    div[data-testid="stFileUploaderDropzone"],
    section[data-testid="stFileUploaderDropzone"] {
        background-color: #FAFCFB !important;
        border: 1.5px dashed #CBD5E1 !important;
        border-radius: 8px !important;
        padding: 1rem 1.2rem !important;
        transition: border-color 0.15s ease, background-color 0.15s ease !important;
    }

    div[data-testid="stFileUploaderDropzone"]:hover,
    section[data-testid="stFileUploaderDropzone"]:hover {
        background-color: #F0F9F8 !important;
        border-color: var(--ss-teal-accent) !important;
    }

    div[data-testid="stFileUploaderDropzoneInstructions"],
    div[data-testid="stFileUploaderDropzoneInstructions"] * {
        color: var(--ss-text-secondary) !important;
    }

    div[data-testid="stFileUploaderDropzone"] button {
        background-color: #FFFFFF !important;
        border: 1px solid var(--ss-border) !important;
        color: var(--ss-navy-deep) !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        box-shadow: var(--ss-shadow-sm) !important;
    }

    div[data-testid="stFileUploaderDropzone"] button:hover {
        background-color: var(--ss-teal-soft) !important;
        border-color: var(--ss-teal-accent) !important;
        color: var(--ss-teal-accent) !important;
    }

    div[data-testid="stFileUploaderDropzone"] button * {
        color: inherit !important;
    }

    div[data-testid="stFileUploaderFile"] {
        background-color: #F8FAF9 !important;
        border: 1px solid var(--ss-border) !important;
        border-radius: 6px !important;
        padding: 6px 10px !important;
    }

    div[data-testid="stFileUploaderFile"] * {
        color: var(--ss-text-primary) !important;
    }

    /* 7. Dropdown Selectbox & Popovers (Eliminates Black-on-Black) */
    div[data-testid="stSelectbox"] > div,
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid var(--ss-border) !important;
        border-radius: 6px !important;
        color: var(--ss-text-primary) !important;
        min-height: 38px !important;
    }

    div[data-baseweb="select"] * {
        color: var(--ss-text-primary) !important;
    }

    /* Injected Dropdown Menu */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"],
    ul[role="listbox"] {
        background-color: #FFFFFF !important;
        border: 1px solid var(--ss-border) !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 18px rgba(20, 40, 61, 0.12) !important;
        padding: 4px !important;
        z-index: 9999 !important;
    }

    li[role="option"],
    div[role="option"] {
        background-color: #FFFFFF !important;
        color: var(--ss-text-primary) !important;
        font-size: 0.82rem !important;
        border-radius: 5px !important;
        padding: 8px 12px !important;
        transition: background 0.1s ease !important;
    }

    li[role="option"] *,
    div[role="option"] * {
        color: var(--ss-text-primary) !important;
    }

    li[role="option"]:hover,
    li[role="option"][aria-selected="true"],
    div[role="option"]:hover,
    div[role="option"][aria-selected="true"] {
        background-color: var(--ss-teal-soft) !important;
        color: var(--ss-navy-deep) !important;
        font-weight: 600 !important;
    }

    li[role="option"]:hover *,
    li[role="option"][aria-selected="true"] *,
    div[role="option"]:hover *,
    div[role="option"][aria-selected="true"] * {
        color: var(--ss-navy-deep) !important;
    }

    /* 8. Tabs Styling */
    div[data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid var(--ss-border) !important;
        gap: 6px !important;
        margin-bottom: 1rem !important;
    }

    button[data-baseweb="tab"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: var(--ss-text-secondary) !important;
        padding: 0.45rem 0.95rem !important;
        border-radius: 6px 6px 0 0 !important;
        background: transparent !important;
        border: none !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--ss-navy-deep) !important;
        border-bottom: 2px solid var(--ss-teal-accent) !important;
        font-weight: 700 !important;
    }

    /* 9. Compact Top Header Bar */
    .ss-top-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.55rem 1rem;
        background: #FFFFFF;
        border: 1px solid var(--ss-border);
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: var(--ss-shadow-sm);
        gap: 12px;
    }

    .ss-header-left {
        display: flex;
        align-items: baseline;
        gap: 8px;
        flex-wrap: wrap;
        flex: 1 1 auto;
        min-width: 0;
    }

    .ss-breadcrumb {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--ss-text-muted);
        display: flex;
        align-items: center;
        gap: 5px;
        white-space: nowrap;
    }

    .ss-header-title {
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        color: var(--ss-navy-deep) !important;
        margin: 0 !important;
        line-height: 1.2 !important;
        white-space: nowrap;
    }

    .ss-header-desc {
        font-size: 0.76rem !important;
        color: var(--ss-text-secondary) !important;
        margin: 0 !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .ss-status-group {
        display: flex;
        align-items: center;
        gap: 6px;
        flex-shrink: 0;
        white-space: nowrap;
    }

    .ss-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 8px;
        border-radius: 999px;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        white-space: nowrap;
    }

    .ss-status-pill.active {
        background: var(--ss-teal-soft);
        border: 1px solid rgba(19, 168, 158, 0.3);
        color: var(--ss-teal-accent);
    }

    .ss-status-pill.inactive {
        background: #F1F4F5;
        border: 1px solid #DCE2E6;
        color: var(--ss-text-muted);
    }

    /* 10. Standard Card System */
    .ss-card {
        background: var(--ss-bg-surface);
        border: 1px solid var(--ss-border);
        border-radius: 9px;
        padding: 1.1rem 1.2rem;
        box-shadow: var(--ss-shadow-sm);
        margin-bottom: 0.85rem;
    }

    .ss-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.70rem;
        padding-bottom: 0.50rem;
        border-bottom: 1px solid var(--ss-border-subtle);
    }

    .ss-card-title {
        font-size: 0.92rem;
        font-weight: 700;
        color: var(--ss-navy-deep);
        margin: 0;
        letter-spacing: -0.015em;
    }

    .ss-card-subtitle {
        font-size: 0.75rem;
        color: var(--ss-text-secondary);
        margin: 1px 0 0 0;
    }

    /* 11. Selective Dark Audio Console */
    .ss-dark-console {
        background: var(--ss-bg-dark);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 2px solid var(--ss-teal-accent);
        border-radius: 9px;
        padding: 1.1rem 1.2rem;
        color: #F1F5F9;
        box-shadow: 0 4px 18px rgba(14, 23, 36, 0.16);
        margin-bottom: 0.85rem;
    }

    .ss-dark-console .ss-card-title {
        color: #FFFFFF !important;
        font-size: 0.92rem;
    }

    .ss-dark-console .ss-card-subtitle {
        color: #94A3B8 !important;
        font-size: 0.75rem;
    }

    .ss-dark-console .ss-card-header {
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* 12. Hero Section */
    .ss-hero-compact {
        background: #FFFFFF;
        border: 1px solid var(--ss-border);
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        box-shadow: var(--ss-shadow-sm);
        margin-bottom: 0.9rem;
    }

    .ss-hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 2px 7px;
        border-radius: 4px;
        background: var(--ss-teal-soft);
        border: 1px solid rgba(19, 168, 158, 0.25);
        color: var(--ss-teal-accent);
        font-size: 0.66rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.45rem;
    }

    .ss-hero-h1 {
        font-size: 1.45rem !important;
        font-weight: 800 !important;
        line-height: 1.2 !important;
        color: var(--ss-navy-deep) !important;
        letter-spacing: -0.03em !important;
        margin: 0 0 0.35rem 0 !important;
    }

    .ss-hero-lead-compact {
        font-size: 0.85rem !important;
        line-height: 1.42 !important;
        color: var(--ss-text-secondary) !important;
        margin-bottom: 0 !important;
    }

    /* 13. Three Coordinated Engine Cards */
    .ss-engine-card {
        background: #FFFFFF;
        border: 1px solid var(--ss-border);
        border-radius: 8px;
        padding: 0.95rem 1rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: var(--ss-shadow-sm);
        border-top: 3px solid transparent;
        transition: transform 0.12s ease, border-color 0.12s ease;
    }

    .ss-engine-card:hover {
        border-color: #CBD5E1;
        transform: translateY(-1px);
    }

    .ss-engine-card.physics { border-top-color: #0284C7; }
    .ss-engine-card.biology { border-top-color: #8B5CF6; }
    .ss-engine-card.aasist { border-top-color: var(--ss-teal-accent); }

    .ss-engine-tag {
        display: inline-block;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.66rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }

    .ss-engine-tag.physics { background: #E0F2FE; color: #0369A1; }
    .ss-engine-tag.biology { background: #F3E8FF; color: #6D28D9; }
    .ss-engine-tag.aasist { background: var(--ss-teal-soft); color: var(--ss-teal-accent); }

    .ss-feature-chip {
        display: inline-flex;
        align-items: center;
        padding: 2px 6px;
        border-radius: 4px;
        background: #F4F6F7;
        color: var(--ss-text-secondary);
        font-size: 0.68rem;
        font-weight: 500;
        margin: 2px 2px 0 0;
    }

    /* 14. Structured Benchmark Summary Strip */
    .ss-benchmark-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-bottom: 0.85rem;
    }

    .ss-bench-cell {
        background: #FFFFFF;
        border: 1px solid var(--ss-border);
        border-radius: 8px;
        padding: 0.75rem 0.95rem;
        box-shadow: var(--ss-shadow-sm);
        display: flex;
        flex-direction: column;
    }

    .ss-bench-label {
        font-size: 0.66rem;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--ss-text-muted);
        margin-bottom: 2px;
    }

    .ss-bench-val {
        font-size: 1.1rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        color: var(--ss-navy-deep);
        line-height: 1.15;
    }

    /* 15. Outcome Verdict Banner */
    .ss-verdict-banner {
        border-radius: 8px;
        padding: 0.95rem 1.15rem;
        margin-bottom: 0.85rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border: 1px solid transparent;
    }

    .ss-verdict-banner.bona-fide {
        background: linear-gradient(135deg, rgba(19, 168, 158, 0.08) 0%, #FFFFFF 100%);
        border-color: rgba(19, 168, 158, 0.35);
    }

    .ss-verdict-banner.inconclusive {
        background: linear-gradient(135deg, rgba(217, 154, 53, 0.08) 0%, #FFFFFF 100%);
        border-color: rgba(217, 154, 53, 0.35);
    }

    .ss-verdict-banner.synthetic {
        background: linear-gradient(135deg, rgba(217, 87, 87, 0.08) 0%, #FFFFFF 100%);
        border-color: rgba(217, 87, 87, 0.35);
    }

    .ss-verdict-title {
        font-size: 1.08rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
        letter-spacing: -0.015em !important;
    }

    .ss-verdict-desc {
        font-size: 0.78rem !important;
        color: var(--ss-text-secondary) !important;
        margin: 2px 0 0 0 !important;
    }

    /* 16. Technical Pills & Badges */
    .ss-spec-pill {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.70rem;
        padding: 2px 6px;
        background: #F4F6F7;
        border: 1px solid var(--ss-border);
        border-radius: 4px;
        color: var(--ss-navy-deep);
    }

    /* 17. Persistent Ethical Footer */
    .ss-footer-note {
        padding: 0.75rem 1.05rem;
        background: #FFFFFF;
        border: 1px solid var(--ss-border);
        border-radius: 8px;
        font-size: 0.74rem;
        color: var(--ss-text-secondary);
        line-height: 1.45;
        margin-top: 1.4rem;
        box-shadow: var(--ss-shadow-sm);
    }
</style>
"""
