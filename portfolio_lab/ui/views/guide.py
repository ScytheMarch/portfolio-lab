"""Concept Guide page — educational reference for every financial concept in Portfolio Lab."""

from __future__ import annotations

import streamlit as st

from portfolio_lab.ui.styles import gradient_header, section_header, CATEGORY_COLORS, GRAY
from portfolio_lab.ui.definitions import (
    DEFINITIONS, all_categories, get_by_category, render_definition_card,
)


def render() -> None:
    st.markdown(gradient_header("Concept Guide", "📖"), unsafe_allow_html=True)
    st.caption(
        "Plain-English reference for every financial concept used in Portfolio Lab. "
        "What it measures, why it matters, how to interpret it, and the formula behind it."
    )

    # Quick legend
    st.markdown(
        '<div style="display:flex;gap:20px;margin-bottom:16px;font-size:0.85em;color:#9ca3af">'
        '<span><span style="color:#22c55e;font-weight:700">Green</span> = good / strong</span>'
        '<span><span style="color:#eab308;font-weight:700">Yellow</span> = adequate / moderate</span>'
        '<span><span style="color:#ef4444;font-weight:700">Red</span> = poor / weak</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Search
    search = st.text_input(
        "Search concepts",
        placeholder="e.g. Sharpe, alpha, Monte Carlo, diversification...",
        help="Filter by concept name, category, or keyword.",
    )

    categories = all_categories()

    for cat in categories:
        entries = get_by_category(cat)
        if not entries:
            continue

        # Filter by search
        if search:
            search_lower = search.lower()
            filtered = {
                k: v for k, v in entries.items()
                if search_lower in k.lower()
                or search_lower in v.get("short", "").lower()
                or search_lower in v.get("long", "").lower()
                or search_lower in v.get("category", "").lower()
            }
            if not filtered:
                continue
            entries = filtered

        cat_color = CATEGORY_COLORS.get(cat, GRAY)

        with st.expander(
            f"**{cat}** ({len(entries)} concepts)",
            expanded=bool(search),
        ):
            for key in sorted(entries.keys()):
                st.markdown(render_definition_card(key), unsafe_allow_html=True)
                st.markdown("")  # spacing
