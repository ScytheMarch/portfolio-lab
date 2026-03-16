"""
portfolio_lab — Portfolio Construction, Optimization & Simulation Platform

Entry point for the Streamlit application.
Run: streamlit run portfolio_lab/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure package is importable when running via streamlit
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── View functions (lazy imports to avoid Streamlit auto-page discovery) ──

def builder_page():
    from portfolio_lab.ui.views.builder import render
    render()

def analysis_page():
    from portfolio_lab.ui.views.analysis import render
    render()

def diagnostics_page():
    from portfolio_lab.ui.views.diagnostics import render
    render()

def guide_page():
    from portfolio_lab.ui.views.guide import render
    render()

# ── Navigation ────────────────────────────────────────────────────────────
_page_builder = st.Page(builder_page, title="Portfolio Builder", icon="🏗️", default=True)
_page_analysis = st.Page(analysis_page, title="Analysis", icon="📊")
_page_diagnostics = st.Page(diagnostics_page, title="Diagnostics", icon="🔬")
_page_guide = st.Page(guide_page, title="Guide", icon="📖")

_all_pages = [_page_builder, _page_analysis, _page_diagnostics, _page_guide]
pg = st.navigation(_all_pages, position="hidden")

# Store page refs for switch_page()
st.session_state["_pages"] = {
    "builder": _page_builder,
    "analysis": _page_analysis,
    "diagnostics": _page_diagnostics,
    "guide": _page_guide,
}

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Inject premium theme ─────────────────────────────────────────────────
from portfolio_lab.ui.styles import inject_theme, gradient_header
inject_theme()

# ── Top navigation bar ───────────────────────────────────────────────────
st.markdown('<div class="top-nav-bar">', unsafe_allow_html=True)

_nav_col_title, _nav_col_links, _nav_col_action = st.columns([1.5, 7.5, 1])

with _nav_col_title:
    st.markdown(
        '<div style="font-size:0.95em;font-weight:800;padding:4px 0;white-space:nowrap">'
        '<span style="background:linear-gradient(135deg,#818cf8,#a78bfa);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent">'
        '📊 Portfolio Lab</span></div>',
        unsafe_allow_html=True,
    )

with _nav_col_links:
    _page_labels = ["Builder", "Analysis", "Diagnostics", "Guide"]
    _page_icons = ["🏗️", "📊", "🔬", "📖"]
    _page_refs = [_page_builder, _page_analysis, _page_diagnostics, _page_guide]
    _btn_cols = st.columns(len(_page_labels))
    for _i, (_label, _icon, _pref) in enumerate(zip(_page_labels, _page_icons, _page_refs)):
        with _btn_cols[_i]:
            if st.button(_label, key=f"nav_{_i}", icon=_icon,
                        use_container_width=True, type="secondary"):
                st.switch_page(_pref)

with _nav_col_action:
    if st.button("Run Analysis", key="nav_run", type="primary",
                 use_container_width=True, icon="🚀"):
        st.session_state["trigger_analysis"] = True
        st.switch_page(_page_analysis)

st.markdown('</div>', unsafe_allow_html=True)

# ── Run ───────────────────────────────────────────────────────────────────
pg.run()
