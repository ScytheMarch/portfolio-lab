"""Design tokens, color constants, and global CSS for Portfolio Lab's premium theme."""

import streamlit as st

# ── Traffic-light colors ─────────────────────────────────────────────────
GREEN = "#22c55e"
YELLOW = "#eab308"
RED = "#ef4444"
GRAY = "#6b7280"
BLUE = "#3b82f6"

# ── Design Tokens ────────────────────────────────────────────────────────
TEXT_PRIMARY = "#f1f5f9"
TEXT_SECONDARY = "#94a3b8"
TEXT_MUTED = "#64748b"
TEXT_DIM = "#475569"

ACCENT_INDIGO = "#6366f1"
ACCENT_VIOLET = "#8b5cf6"

CARD_BG = "linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015))"
CARD_BORDER = "1px solid rgba(255,255,255,0.08)"
CARD_RADIUS = "12px"
CARD_PADDING = "14px 18px"
CARD_PADDING_LG = "20px 24px"

SPACE_XS = "4px"
SPACE_SM = "8px"
SPACE_MD = "12px"
SPACE_LG = "16px"
SPACE_XL = "24px"

# ── Category colors ──────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "Portfolio Theory": "#6366f1",
    "Risk Metrics": "#ef4444",
    "Factor Models": "#8b5cf6",
    "Simulation": "#3b82f6",
    "Income": "#22c55e",
    "Optimization": "#f97316",
}


def glass_card(content: str, border_color: str = "rgba(255,255,255,0.08)",
               padding: str = CARD_PADDING) -> str:
    """Wrap HTML content in a frosted-glass card div."""
    return (
        f'<div style="background:{CARD_BG};border:1px solid {border_color};'
        f'border-radius:{CARD_RADIUS};padding:{padding};'
        f'backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);'
        f'box-shadow:0 2px 8px rgba(0,0,0,0.15);margin-bottom:12px'
        f'">{content}</div>'
    )


def section_header(text: str) -> str:
    """Return styled uppercase section header HTML."""
    return (
        f'<div style="color:{TEXT_MUTED};font-size:0.75em;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:1.5px;'
        f'margin:20px 0 12px 0">{text}</div>'
    )


def gradient_header(text: str, icon: str = "") -> str:
    """Return a gradient-accented header."""
    prefix = f"{icon} " if icon else ""
    return (
        f'<h2 style="font-weight:700;letter-spacing:-0.5px;margin-bottom:4px">'
        f'{prefix}<span style="background:linear-gradient(135deg,#818cf8,#a78bfa);'
        f'-webkit-background-clip:text;-webkit-text-fill-color:transparent">'
        f'{text}</span></h2>'
    )


def metric_badge(label: str, value: str, color: str = GREEN) -> str:
    """Small colored badge for inline metric display."""
    return (
        f'<span style="display:inline-block;background:rgba({_hex_to_rgb(color)},0.15);'
        f'color:{color};padding:4px 10px;border-radius:8px;font-size:0.85em;'
        f'font-weight:600;margin-right:8px;border:1px solid rgba({_hex_to_rgb(color)},0.3)">'
        f'{label}: {value}</span>'
    )


def ticker_chip(ticker: str, color: str = ACCENT_INDIGO) -> str:
    """Render a single ticker as a styled chip."""
    return (
        f'<span style="display:inline-block;background:rgba({_hex_to_rgb(color)},0.15);'
        f'color:{color};padding:3px 10px;border-radius:16px;font-size:0.8em;'
        f'font-weight:600;margin:2px 4px 2px 0;border:1px solid rgba({_hex_to_rgb(color)},0.25)">'
        f'{ticker}</span>'
    )


def ticker_chips_html(tickers: list[str]) -> str:
    """Render a list of tickers as styled chips."""
    return '<div style="display:flex;flex-wrap:wrap;gap:4px;margin:8px 0">' + \
        "".join(ticker_chip(t) for t in tickers) + "</div>"


def quality_color(value: float, thresholds: dict) -> str:
    """Return GREEN/YELLOW/RED based on value and threshold dict."""
    if "excellent" in thresholds and value >= thresholds["excellent"]:
        return GREEN
    if "good" in thresholds and value >= thresholds["good"]:
        return GREEN
    if "adequate" in thresholds and value >= thresholds["adequate"]:
        return YELLOW
    return RED


def _hex_to_rgb(hex_color: str) -> str:
    """Convert #RRGGBB to 'R,G,B' string for rgba()."""
    h = hex_color.lstrip("#")
    return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"


# ── Global CSS ───────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
/* ── Import modern font ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global foundation ──────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Rich textured background ─────────────────────────────────────── */
.stApp {
    background:
        radial-gradient(ellipse at 15% 10%, rgba(99, 102, 241, 0.06) 0%, transparent 45%),
        radial-gradient(ellipse at 85% 80%, rgba(139, 92, 246, 0.04) 0%, transparent 40%),
        radial-gradient(ellipse at 50% 50%, rgba(30, 27, 46, 0.8) 0%, transparent 70%),
        linear-gradient(145deg, #08080f 0%, #0c0b14 30%, #10101c 60%, #0a0a12 100%) !important;
}

/* Subtle noise grain overlay */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        repeating-conic-gradient(rgba(255,255,255,0.008) 0% 25%, transparent 0% 50%) 0 0 / 3px 3px,
        radial-gradient(ellipse at 20% 60%, rgba(99, 102, 241, 0.035) 0%, transparent 50%),
        radial-gradient(ellipse at 75% 25%, rgba(168, 85, 247, 0.025) 0%, transparent 45%);
    pointer-events: none;
    z-index: 0;
}

/* ── Hide sidebar ───────────────────────────────────────────────────── */
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
header[data-testid="stHeader"] { display: none !important; }
.stDeployButton { display: none !important; }

/* ── Headers ────────────────────────────────────────────────────────── */
.stMarkdown h1, .stMarkdown h2 {
    font-weight: 700 !important;
    letter-spacing: -0.3px;
}
.stMarkdown h3 {
    font-weight: 600 !important;
    letter-spacing: -0.2px;
}

/* ── Metric cards — frosted glass ──────────────────────────────────── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg,
        rgba(255,255,255,0.05) 0%,
        rgba(255,255,255,0.02) 100%) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.04) !important;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(99, 102, 241, 0.25) !important;
    background: linear-gradient(135deg,
        rgba(99, 102, 241, 0.08) 0%,
        rgba(139, 92, 246, 0.04) 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3),
                0 0 20px rgba(99, 102, 241, 0.08),
                inset 0 1px 0 rgba(255, 255, 255, 0.06) !important;
}
[data-testid="stMetric"] label {
    font-size: 0.7em !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #64748b !important;
    font-weight: 600 !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-weight: 700 !important;
    font-size: 1.8em !important;
    color: #f1f5f9 !important;
}

/* ── Buttons (secondary) ──────────────────────────────────────────── */
.stButton > button[kind="secondary"],
.stButton > button:not([kind]) {
    background: linear-gradient(135deg,
        rgba(255,255,255,0.04) 0%,
        rgba(255,255,255,0.02) 100%) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.2px;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    padding: 8px 20px !important;
    backdrop-filter: blur(8px) !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.15) !important;
}
.stButton > button[kind="secondary"]:hover,
.stButton > button:not([kind]):hover {
    background: linear-gradient(135deg,
        rgba(99, 102, 241, 0.12) 0%,
        rgba(139, 92, 246, 0.06) 100%) !important;
    border-color: rgba(99, 102, 241, 0.3) !important;
    color: #c7d2fe !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
}

/* ── Buttons (primary) ────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #7c3aed 50%, #8b5cf6 100%) !important;
    border: 1px solid rgba(129, 140, 248, 0.4) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.2px;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    padding: 8px 20px !important;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.25),
                0 2px 8px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #818cf8 0%, #8b5cf6 50%, #a78bfa 100%) !important;
    box-shadow: 0 0 30px rgba(99, 102, 241, 0.35),
                0 4px 15px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0px) scale(0.98); }

/* ── Selectboxes / Inputs ─────────────────────────────────────────── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input,
.stTextArea > div > textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.25s ease !important;
    backdrop-filter: blur(8px) !important;
    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.2) !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div > input:focus,
.stTextInput > div > div > input:focus,
.stTextArea > div > textarea:focus {
    border-color: rgba(99, 102, 241, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1),
                inset 0 1px 3px rgba(0, 0, 0, 0.2) !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(255,255,255,0.02);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.06);
    backdrop-filter: blur(8px);
    position: sticky;
    top: 0;
    z-index: 99;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    padding: 8px 16px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(99, 102, 241, 0.15) !important;
    box-shadow: 0 0 10px rgba(99, 102, 241, 0.1) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    overflow-anchor: none;
    padding-top: 8px !important;
}
[data-stale="false"] { scroll-margin-top: 0 !important; }
.stTabs { scroll-margin-top: 0 !important; overflow-anchor: none; }

/* ── Expanders — frosted glass ────────────────────────────────────── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.25s ease !important;
    backdrop-filter: blur(8px) !important;
}
.streamlit-expanderHeader:hover {
    background: rgba(99, 102, 241, 0.06) !important;
    border-color: rgba(99, 102, 241, 0.18) !important;
}
details {
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.015) !important;
    transition: all 0.25s ease !important;
    backdrop-filter: blur(8px) !important;
}
details:hover { border-color: rgba(99, 102, 241, 0.15) !important; }
details[open] {
    border-color: rgba(99, 102, 241, 0.2) !important;
    background: rgba(99, 102, 241, 0.025) !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15) !important;
}

/* ── Plotly charts — glass container ──────────────────────────────── */
.js-plotly-plot .plotly .main-svg { border-radius: 12px; }
[data-testid="stPlotlyChart"] {
    background: linear-gradient(135deg,
        rgba(255,255,255,0.02) 0%,
        rgba(255,255,255,0.008) 100%) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    padding: 8px !important;
    backdrop-filter: blur(8px) !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15) !important;
    transition: all 0.25s ease !important;
}
[data-testid="stPlotlyChart"]:hover {
    border-color: rgba(99, 102, 241, 0.12) !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
}

/* ── Progress bar ──────────────────────────────────────────────────── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #a78bfa) !important;
    border-radius: 10px !important;
    box-shadow: 0 0 10px rgba(99, 102, 241, 0.3) !important;
}

/* ── Dividers ──────────────────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.05) !important; margin: 16px 0 !important; }

/* ── Dataframes ────────────────────────────────────────────────────── */
.stDataFrame {
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    overflow: hidden;
    backdrop-filter: blur(8px) !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15) !important;
}

/* ── Alerts ────────────────────────────────────────────────────────── */
.stAlert {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
}

/* ── Scrollbar ─────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.2); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99, 102, 241, 0.4); }

/* ── Smooth transitions ────────────────────────────────────────────── */
a, button, input, select, details, [role="tab"] {
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

/* ── Captions ──────────────────────────────────────────────────────── */
.stCaption, [data-testid="stCaptionContainer"] {
    color: #475569 !important;
    font-size: 0.8em !important;
    letter-spacing: 0.3px;
}

/* ── Main content ──────────────────────────────────────────────────── */
.stMainBlockContainer { padding-top: 0.5rem !important; }

/* ── Top nav bar ───────────────────────────────────────────────────── */
.top-nav-bar {
    background: linear-gradient(135deg,
        rgba(15, 15, 25, 0.95) 0%,
        rgba(20, 18, 35, 0.92) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(99, 102, 241, 0.15);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    padding: 8px 0 4px 0;
    margin: -1rem -1rem 12px -1rem;
}
.top-nav-bar .stButton > button {
    padding: 6px 8px !important;
    font-size: 0.76em !important;
    white-space: nowrap !important;
    min-height: 38px !important;
    max-height: 38px !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    gap: 4px !important;
    border-radius: 10px !important;
    letter-spacing: 0.2px;
}

/* ── Radio buttons compact ─────────────────────────────────────────── */
.stRadio > div { gap: 0.3rem !important; }

/* ── Multiselect tags ──────────────────────────────────────────────── */
.stMultiSelect [data-baseweb="tag"] {
    background: rgba(99, 102, 241, 0.15) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 8px !important;
}

/* ── Column containers ─────────────────────────────────────────────── */
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    transition: all 0.25s ease !important;
}
</style>
"""


def inject_theme():
    """Inject the global CSS theme into the Streamlit app."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    # Prevent scroll-jump on tab click
    st.markdown("""
    <script>
    (function() {
        const orig = Element.prototype.scrollIntoView;
        Element.prototype.scrollIntoView = function(opts) {
            if (this.closest && this.closest('[data-baseweb="tab-panel"]')) return;
            if (this.closest && this.closest('.stTabs')) return;
            return orig.call(this, opts);
        };
    })();
    </script>
    """, unsafe_allow_html=True)
