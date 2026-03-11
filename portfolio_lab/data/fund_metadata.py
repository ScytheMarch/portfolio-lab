"""
Fund metadata extraction: expense ratios, yields, categories, fund type.

Uses yfinance .info dict as primary source. Degrades gracefully with warnings.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class FundInfo:
    """Metadata for a single fund/ETF/ticker."""
    ticker: str
    name: str = ""
    category: str = "Unknown"
    asset_class: str = "Unknown"
    expense_ratio: Optional[float] = None  # decimal, e.g. 0.0003 for 0.03%
    dividend_yield: Optional[float] = None  # trailing 12m yield, decimal
    ytd_return: Optional[float] = None
    total_assets: Optional[float] = None
    fund_family: str = ""
    quote_type: str = ""  # ETF, MUTUALFUND, EQUITY, etc.
    warnings: list[str] = field(default_factory=list)

    @property
    def expense_ratio_display(self) -> str:
        if self.expense_ratio is not None:
            return f"{self.expense_ratio * 100:.2f}%"
        return "N/A (not available)"

    @property
    def is_expense_ratio_missing(self) -> bool:
        return self.expense_ratio is None


def fetch_fund_metadata(tickers: list[str]) -> dict[str, FundInfo]:
    """
    Fetch metadata for each ticker using yfinance .info.

    Returns:
        Dict mapping ticker -> FundInfo.
        Missing fields are flagged in FundInfo.warnings.
    """
    result = {}
    for t in tickers:
        info_dict = _safe_fetch_info(t)
        fund = _parse_fund_info(t, info_dict)
        result[t] = fund
    return result


def _safe_fetch_info(ticker: str) -> dict:
    """Fetch yfinance info with error handling."""
    try:
        obj = yf.Ticker(ticker)
        info = obj.info or {}
        return info
    except Exception as e:
        logger.warning(f"Failed to fetch info for {ticker}: {e}")
        return {}


def _parse_fund_info(ticker: str, info: dict) -> FundInfo:
    """Parse yfinance info dict into FundInfo with explicit warnings."""
    warnings = []
    fund = FundInfo(ticker=ticker)

    fund.name = info.get("longName") or info.get("shortName") or ticker
    fund.quote_type = info.get("quoteType", "")
    fund.fund_family = info.get("fundFamily", "")

    # Category
    fund.category = info.get("category") or _infer_category(info)

    # Asset class inference
    fund.asset_class = _infer_asset_class(fund.category, fund.quote_type, ticker)

    # Expense ratio
    er = info.get("annualReportExpenseRatio")
    if er is not None and isinstance(er, (int, float)) and er >= 0:
        fund.expense_ratio = float(er)
    else:
        # Try alternate field names
        er2 = info.get("expenseRatio")
        if er2 is not None and isinstance(er2, (int, float)) and er2 >= 0:
            fund.expense_ratio = float(er2)
        else:
            warnings.append(
                f"Expense ratio not available for {ticker}. "
                "Manual input recommended. Do NOT assume zero."
            )

    # Dividend yield
    dy = info.get("yield") or info.get("dividendYield") or info.get("trailingAnnualDividendYield")
    if dy is not None and isinstance(dy, (int, float)):
        fund.dividend_yield = float(dy)
    else:
        warnings.append(f"Dividend yield not available for {ticker}.")

    # Total assets
    ta = info.get("totalAssets")
    if ta is not None:
        fund.total_assets = float(ta)

    fund.warnings = warnings
    for w in warnings:
        logger.warning(w)

    return fund


def _infer_category(info: dict) -> str:
    """Best-effort category inference from info dict."""
    sector = info.get("sector", "")
    qt = info.get("quoteType", "")
    if qt == "ETF":
        return info.get("category", "ETF - Uncategorized")
    if qt == "MUTUALFUND":
        return info.get("category", "Mutual Fund - Uncategorized")
    if sector:
        return f"Equity - {sector}"
    return "Unknown"


def _infer_asset_class(category: str, quote_type: str, ticker: str) -> str:
    """Infer broad asset class from category string and ticker heuristics."""
    cat_lower = category.lower()
    ticker_upper = ticker.upper()

    # Treasury / fixed income proxies
    treasury_tickers = {"BIL", "SHV", "SHY", "IEF", "TLT", "TIP", "GOVT", "VGSH", "VGIT", "VGLT"}
    bond_keywords = {"bond", "fixed income", "treasury", "government", "aggregate", "income"}
    if ticker_upper in treasury_tickers:
        return "Treasury/Government Bond"
    if any(kw in cat_lower for kw in bond_keywords):
        return "Fixed Income"

    # Equity
    equity_keywords = {"stock", "equity", "large", "mid", "small", "growth", "value", "blend"}
    if any(kw in cat_lower for kw in equity_keywords):
        return "Equity"

    # Real estate
    if "real estate" in cat_lower or "reit" in cat_lower:
        return "Real Estate"

    # Commodities
    commodity_tickers = {"GLD", "IAU", "SLV", "DBC", "GSG", "PDBC"}
    if ticker_upper in commodity_tickers or "commodity" in cat_lower:
        return "Commodity"

    # International
    if "international" in cat_lower or "foreign" in cat_lower or "emerging" in cat_lower:
        return "International Equity"

    return "Other"
