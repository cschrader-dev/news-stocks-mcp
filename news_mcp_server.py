"""
News & Stocks MCP Server (Hardened)
====================================

A production-grade Model Context Protocol server that provides AI agents with
structured access to global news and finance/stock news from free, publicly
accessible RSS sources.

This server exposes three MCP primitives:

    TOOLS      – Fetch live news data (global search, per-ticker, portfolio snapshot).
    RESOURCES  – Read-only access to local JSON files (portfolio config, cached snapshots).
    PROMPTS    – Reusable LLM prompt templates for trade-idea generation and daily briefs.

The server itself performs NO trading, NO recommendations, and NO financial
advice.  It is a pure *data-access and prompt-template* layer.  The consuming
AI agent / LLM is responsible for all reasoning and must treat its own output
as non-financial-advice by default.

Dependencies (see requirements.txt):
    fastmcp >= 2.0.0
    httpx   >= 0.27.0
    feedparser >= 6.0.0
    pydantic >= 2.0.0

Run:
    python news_mcp_server.py          # stdio transport (default)
    python news_mcp_server.py --help   # see FastMCP CLI options
"""

from __future__ import annotations

import json
import logging
import os
import re
import textwrap
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

# pyrefly: ignore [missing-import]
import feedparser
import httpx
# pyrefly: ignore [missing-import]
from fastmcp import FastMCP
# pyrefly: ignore [missing-import]
from fastmcp.exceptions import ToolError
# pyrefly: ignore [missing-import]
from fastmcp.prompts import Message
from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION (all from environment variables, safe defaults provided)
# ═══════════════════════════════════════════════════════════════════════════════

NEWS_SEARCH_URL: str = os.getenv(
    "NEWS_SEARCH_URL",
    "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en",
)
STOCK_NEWS_URL: str = os.getenv(
    "STOCK_NEWS_URL",
    "https://news.google.com/rss/search?q={symbol}+stock+market&hl=en-US&gl=US&ceid=US:en",
)
HTTP_TIMEOUT: float = float(os.getenv("HTTP_TIMEOUT", "15.0"))
MAX_LIMIT: int = int(os.getenv("MAX_LIMIT", "50"))
DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(Path(__file__).parent / "data")))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("news_mcp_server")

# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS (shared return schemas)
# ═══════════════════════════════════════════════════════════════════════════════


class Article(BaseModel):
    """A single normalised news article returned by every news-fetching tool.

    Fields:
        title:     Headline of the article.
        summary:   Short abstract / description (may be None if the feed omits it).
        link:      Canonical URL to the full article.
        published: ISO-8601-ish publication timestamp as provided by the feed
                   (may be None if the feed omits it).
        source:    Human-readable name of the originating publisher (may be None).
    """

    title: str = Field(..., description="Headline of the article.")
    summary: str | None = Field(
        None, description="Short abstract or description of the article."
    )
    link: str = Field(..., description="Canonical URL to the full article.")
    published: str | None = Field(
        None,
        description="Publication timestamp as provided by the feed (ISO-8601 when available).",
    )
    source: str | None = Field(
        None, description="Human-readable name of the originating publisher."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MCP SERVER INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════════

mcp = FastMCP(
    name="News & Stocks MCP (Hardened)",
    instructions=textwrap.dedent("""\
        You are connected to the **News & Stocks MCP Server**, a read-only data
        layer for global and financial news.

        Available capabilities:
        ────────────────────────
        TOOLS
          • get_global_news   – search global news by keyword.
          • get_stock_news    – search stock/finance news by ticker symbol.
          • get_market_snapshot – batch-fetch news for multiple tickers at once.

        RESOURCES
          • data://portfolio            – the user's saved portfolio (tickers, weights, risk prefs).
          • data://daily_news_snapshot   – a cached daily global-news snapshot.

        PROMPTS
          • portfolio_news_recommendations – structured trade-idea prompt template.
          • daily_briefing                 – daily macro/sector briefing prompt template.

        IMPORTANT
          • This server provides DATA and PROMPT TEMPLATES only.
          • It does NOT execute trades, place orders, or perform any stateful mutations.
          • All recommendations the agent produces should be clearly labelled as
            informational and NOT financial advice.
    """),
)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS – HTTP fetching & RSS normalisation
# ═══════════════════════════════════════════════════════════════════════════════

_TICKER_RE = re.compile(r"^[A-Z0-9]{1,10}$")


def _clamp_limit(limit: int) -> int:
    """Clamp *limit* to the range [1, MAX_LIMIT]."""
    return max(1, min(limit, MAX_LIMIT))


def _validate_query(query: str) -> str:
    """Return a stripped query string, raising ToolError if invalid."""
    query = query.strip()
    if len(query) < 2:
        raise ToolError(
            "The 'query' parameter must be a non-empty string of at least "
            "2 characters.  Received: " + repr(query)
        )
    return query


def _validate_symbol(symbol: str) -> str:
    """Return an uppercased, validated ticker symbol."""
    symbol = symbol.strip().upper()
    if not _TICKER_RE.match(symbol):
        raise ToolError(
            f"Invalid ticker symbol '{symbol}'.  Symbols must be 1-10 "
            "uppercase alphanumeric characters (e.g. AAPL, MSFT, NVDA)."
        )
    return symbol


async def _fetch_feed(url: str, context_label: str) -> feedparser.FeedParserDict:
    """Fetch an RSS/Atom feed from *url* and return the parsed result.

    Raises ToolError on network or parsing failures with a human-readable
    message that includes *context_label* (e.g. "global news for 'AI'").
    """
    logger.info("Fetching feed  url=%s  context=%s", url, context_label)
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
    except httpx.TimeoutException:
        msg = f"HTTP timeout ({HTTP_TIMEOUT}s) while fetching {context_label}."
        logger.warning(msg)
        raise ToolError(msg)
    except httpx.HTTPStatusError as exc:
        msg = (
            f"HTTP {exc.response.status_code} error while fetching "
            f"{context_label}: {exc.response.reason_phrase}"
        )
        logger.error(msg)
        raise ToolError(msg)
    except httpx.HTTPError as exc:
        msg = f"Network error while fetching {context_label}: {exc}"
        logger.error(msg)
        raise ToolError(msg)

    feed = feedparser.parse(response.text)
    if feed.bozo and not feed.entries:
        msg = (
            f"RSS/Atom parsing failed for {context_label}.  "
            f"Parser error: {feed.bozo_exception}"
        )
        logger.error(msg)
        raise ToolError(msg)
    return feed


def _normalise_entries(
    feed: feedparser.FeedParserDict, limit: int
) -> list[dict[str, Any]]:
    """Convert feedparser entries into a list of Article-shaped dicts."""
    articles: list[dict[str, Any]] = []
    for entry in feed.entries[:limit]:
        # Extract source – feedparser puts it in 'source.title' or we fall
        # back to the feed-level title.  Some feeds set source as a plain
        # string rather than a dict-like object, so we guard with try/except.
        source_name: str | None = None
        try:
            entry_source = getattr(entry, "source", None)
            if entry_source and isinstance(entry_source, dict):
                source_name = entry_source.get("title")
        except (AttributeError, TypeError):
            pass
        if not source_name:
            source_name = feed.feed.get("title")

        articles.append(
            Article(
                title=entry.get("title", "(no title)"),
                summary=entry.get("summary") or entry.get("description"),
                link=entry.get("link", ""),
                published=entry.get("published") or entry.get("updated"),
                source=source_name,
            ).model_dump()
        )
    return articles


def _read_json_file(filename: str, human_label: str) -> dict[str, Any]:
    """Read and parse a JSON file from DATA_DIR.

    Raises ToolError with a clear message if the file is missing or malformed.
    *human_label* is used in error messages (e.g. "portfolio configuration").
    """
    filepath = DATA_DIR / filename
    # Safety: resolve and ensure the path is inside DATA_DIR.
    resolved = filepath.resolve()
    data_dir_resolved = DATA_DIR.resolve()
    try:
        resolved.relative_to(data_dir_resolved)
    except ValueError:
        raise ToolError(
            f"Refusing to read '{filename}': path escapes the data directory."
        )
    if not resolved.is_file():
        logger.warning("Missing data file: %s", resolved)
        raise ToolError(
            f"The {human_label} file was not found at '{resolved}'.  "
            "Please create it or set the DATA_DIR environment variable."
        )
    try:
        text = resolved.read_text(encoding="utf-8")
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", resolved, exc)
        raise ToolError(
            f"The {human_label} file at '{resolved}' contains invalid JSON: {exc}"
        )
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True,
    },
)
async def get_global_news(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Fetch recent global news articles matching an arbitrary search query.

    This tool queries a free, public RSS news aggregator (Google News RSS by
    default) and returns normalised article objects.  It is the primary
    entry-point for broad, topic-based news retrieval — use it for macro-
    economic, geopolitical, sector, or technology queries.

    The tool performs outbound HTTP only; it does NOT modify any external
    system, place orders, or store data.

    Args:
        query: A free-text search query describing the news topic you want.
            Must be at least 2 characters after trimming whitespace.
            Examples: "Federal Reserve interest rates", "AI chip demand",
            "European energy crisis", "OPEC oil production cuts".
        limit: Maximum number of articles to return.  Clamped to the range
            [1, MAX_LIMIT] (MAX_LIMIT defaults to 50 and is configurable
            via the MAX_LIMIT environment variable).  Defaults to 10.

    Returns:
        A JSON array of article objects, each containing:
            - title (str):      Headline of the article.
            - summary (str|null): Short abstract or description.
            - link (str):       Canonical URL to the full article.
            - published (str|null): Publication timestamp from the feed.
            - source (str|null): Name of the originating publisher.

    Raises:
        ToolError: If the query is invalid, the upstream feed is unreachable,
            or RSS parsing fails.

    Example scenario:
        The agent wants to understand the latest developments around central
        bank policy.  It calls:
            get_global_news(query="central bank interest rate decision", limit=5)
        and receives a list of 5 recent articles it can summarise or cross-
        reference against the user's portfolio.
    """
    query = _validate_query(query)
    limit = _clamp_limit(limit)
    logger.info("TOOL get_global_news  query=%r  limit=%d", query, limit)

    url = NEWS_SEARCH_URL.format(query=quote_plus(query))
    feed = await _fetch_feed(url, f"global news for '{query}'")
    articles = _normalise_entries(feed, limit)

    logger.info("get_global_news returned %d articles", len(articles))
    return articles


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True,
    },
)
async def get_stock_news(symbol: str, limit: int = 10) -> list[dict[str, Any]]:
    """Fetch recent finance / stock-market news for a single ticker symbol.

    This tool queries a free, public RSS news feed scoped to a specific stock
    ticker (e.g. NVDA, AAPL, TSLA) and returns normalised article objects.
    Use it when the agent needs sentiment, catalysts, or event-driven context
    for a particular equity.

    The tool performs outbound HTTP only; it does NOT modify any external
    system, place orders, or store data.

    Args:
        symbol: The stock ticker symbol to search for.  Will be uppercased
            and validated (must be 1–10 alphanumeric characters).
            Examples: "NVDA", "AAPL", "TSLA", "MSFT", "AMZN".
        limit: Maximum number of articles to return.  Clamped to [1, MAX_LIMIT].
            Defaults to 10.

    Returns:
        A JSON array of article objects with the same schema as get_global_news:
            - title, summary, link, published, source.

    Raises:
        ToolError: If the symbol is invalid, the upstream feed is unreachable,
            or RSS parsing fails.

    Example scenario:
        The agent holds TSLA in the portfolio and wants to check for overnight
        catalysts.  It calls:
            get_stock_news(symbol="TSLA", limit=5)
        and receives the 5 most recent TSLA-related articles.
    """
    symbol = _validate_symbol(symbol)
    limit = _clamp_limit(limit)
    logger.info("TOOL get_stock_news  symbol=%s  limit=%d", symbol, limit)

    url = STOCK_NEWS_URL.format(symbol=quote_plus(symbol))
    feed = await _fetch_feed(url, f"stock news for {symbol}")
    articles = _normalise_entries(feed, limit)

    logger.info("get_stock_news(%s) returned %d articles", symbol, len(articles))
    return articles


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
    },
)
async def get_market_snapshot(
    symbols: list[str], limit_per_symbol: int = 5
) -> dict[str, Any]:
    """Fetch a bundled news snapshot for multiple ticker symbols at once.

    This is a convenience tool that calls get_stock_news logic for every
    symbol in the provided list and assembles the results into a single
    dictionary keyed by ticker.  It is ideal for portfolio-wide scans where
    the agent needs to assess news across many positions simultaneously.

    Per-symbol failures are handled gracefully: if one ticker's feed fails,
    an error object is placed under that key and the remaining tickers are
    still returned successfully.

    The tool performs outbound HTTP only; it does NOT modify any external
    system, place orders, or store data.

    Args:
        symbols: A list of stock ticker symbols to fetch news for.
            Each symbol is uppercased and validated independently.
            Must contain at least 1 symbol.
            Example: ["NVDA", "TSLA", "MSFT", "JNJ"]
        limit_per_symbol: Maximum number of articles to return per symbol.
            Clamped to [1, MAX_LIMIT].  Defaults to 5.

    Returns:
        A JSON object mapping each (uppercased) symbol to either:
          - A list of article objects (same schema as get_global_news), OR
          - An error object {"error": "<description>"} if that symbol's fetch failed.

        Example structure:
        {
            "NVDA": [{"title": "...", "summary": "...", ...}, ...],
            "TSLA": [{"title": "...", "summary": "...", ...}, ...],
            "BADTICKER": {"error": "Invalid ticker symbol 'BADTICKER'. ..."}
        }

    Raises:
        ToolError: Only if the entire symbols list is empty.  Individual
            symbol failures are returned inline as error objects.

    Example scenario:
        The agent has loaded the user's portfolio (via the 'portfolio'
        resource) and wants a comprehensive news scan.  It calls:
            get_market_snapshot(
                symbols=["NVDA", "AAPL", "MSFT", "TSLA", "AMZN", "JNJ"],
                limit_per_symbol=3
            )
        and receives a dictionary with up to 3 articles per ticker, which it
        can then feed into the portfolio_news_recommendations prompt.
    """
    if not symbols:
        raise ToolError(
            "The 'symbols' parameter must be a non-empty list of ticker symbols."
        )

    limit_per_symbol = _clamp_limit(limit_per_symbol)
    logger.info(
        "TOOL get_market_snapshot  symbols=%s  limit_per_symbol=%d",
        symbols,
        limit_per_symbol,
    )

    results: dict[str, Any] = {}

    for raw_symbol in symbols:
        try:
            sym = _validate_symbol(raw_symbol)
        except ToolError as exc:
            results[raw_symbol.strip().upper() or raw_symbol] = {
                "error": str(exc)
            }
            continue

        try:
            url = STOCK_NEWS_URL.format(symbol=quote_plus(sym))
            feed = await _fetch_feed(url, f"stock news for {sym}")
            results[sym] = _normalise_entries(feed, limit_per_symbol)
        except ToolError as exc:
            logger.warning("get_market_snapshot: failed for %s: %s", sym, exc)
            results[sym] = {"error": str(exc)}

    logger.info(
        "get_market_snapshot completed for %d symbols (%d succeeded)",
        len(symbols),
        sum(1 for v in results.values() if isinstance(v, list)),
    )
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# RESOURCES
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.resource(
    "data://portfolio",
    name="Portfolio",
    description=textwrap.dedent("""\
        The user's saved portfolio configuration, stored as a local JSON file.

        This resource provides read-only access to the portfolio definition at
        ``data/portfolio.json`` (relative to DATA_DIR).  The file contains:

          • **tickers** – an array of held positions, each with:
              - symbol (str): the ticker symbol (e.g. "NVDA").
              - weight (float): portfolio weight as a decimal (0.0–1.0).
              - risk_preference (str): "conservative", "moderate", or "aggressive".
              - notes (str): free-text investment thesis or context.
          • **name** – a human-readable portfolio name.
          • **description** – a brief description of the portfolio strategy.
          • **last_updated** – ISO-8601 date of the last portfolio edit.

        The agent typically reads this resource FIRST when the user asks for
        portfolio-aware analysis, then passes the tickers into
        ``get_market_snapshot`` or combines it with the
        ``portfolio_news_recommendations`` prompt.

        If the file is missing or contains invalid JSON, a clear error is returned.
    """),
    mime_type="application/json",
)
def read_portfolio() -> str:
    """Read and return the portfolio configuration JSON file."""
    data = _read_json_file("portfolio.json", "portfolio configuration")
    return json.dumps(data, indent=2)


@mcp.resource(
    "data://daily_news_snapshot",
    name="DailyNewsSnapshot",
    description=textwrap.dedent("""\
        A cached daily snapshot of global news articles, stored as a local
        JSON file at ``data/daily_news_snapshot.json``.

        This resource exists so the agent can perform **reproducible** analysis
        on a fixed set of articles (rather than hitting live feeds, which change
        every few minutes).  The file structure includes:

          • **snapshot_date** (str): the date the snapshot represents.
          • **generated_at** (str): ISO-8601 timestamp of when it was captured.
          • **articles** (array): list of article objects, each with:
              - title, summary, link, published, source
              (same schema as the get_global_news tool output).

        The intended workflow is:
          1. A cron job or manual step calls ``get_global_news`` and saves the
             result into ``data/daily_news_snapshot.json``.
          2. The agent reads this resource for its daily briefing and can re-run
             analysis deterministically.

        If the file is missing or contains invalid JSON, a clear error is returned.
    """),
    mime_type="application/json",
)
def read_daily_news_snapshot() -> str:
    """Read and return the daily news snapshot JSON file."""
    data = _read_json_file("daily_news_snapshot.json", "daily news snapshot")
    return json.dumps(data, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.prompt(
    name="portfolio_news_recommendations",
    description=textwrap.dedent("""\
        A detailed, reusable prompt template that instructs an LLM to act as a
        **cautious trading assistant**.  It accepts two JSON payloads — the
        user's portfolio configuration and recent news — and asks the model to:

          1. Map each news article to the portfolio ticker(s) it is most
             relevant to.
          2. Determine per-ticker sentiment (bullish / bearish / neutral).
          3. Propose at most 3 concrete trade ideas with action (buy / sell /
             hold), thesis, risks, and time horizon.
          4. Output a **structured JSON** response with two top-level keys:
             - "per_ticker": per-ticker sentiment, key news, and notes.
             - "ideas": an array of trade ideas.

        This prompt is designed for agents that have already fetched news
        (via get_market_snapshot or the daily_news_snapshot resource) and
        loaded the portfolio (via the portfolio resource).

        DISCLAIMER: The prompt explicitly instructs the model that its output
        is informational only and does NOT constitute financial advice.
    """),
)
def portfolio_news_recommendations(
    portfolio_json: str,
    news_json: str,
) -> list[Message]:
    """Build the portfolio-news-recommendations prompt.

    Args:
        portfolio_json: The full portfolio JSON string (as returned by the
            'data://portfolio' resource).  Must be valid JSON containing at
            least a 'tickers' array.
        news_json: The news JSON string — either the output of
            get_market_snapshot (a dict mapping symbols to article arrays)
            or the daily_news_snapshot resource content.
    """
    system_prompt = textwrap.dedent("""\
        You are a **cautious, analytical trading assistant**.  You have been
        given the user's portfolio configuration and a batch of recent news
        articles.  Your job is to:

        1. **Map news to tickers** — for each article, identify which
           portfolio ticker(s) it is most relevant to.  An article may map
           to zero or multiple tickers.

        2. **Determine sentiment** — for each ticker that has at least one
           relevant article, assess the aggregate sentiment:
           - "bullish" — news is predominantly positive for the stock.
           - "bearish" — news is predominantly negative.
           - "neutral" — mixed or no clear directional signal.

        3. **Propose trade ideas** — suggest AT MOST 3 concrete trade ideas.
           Each idea must include:
           - ticker: the symbol.
           - action: one of "buy", "sell", or "hold".
           - thesis: a concise explanation (2-4 sentences).
           - risks: key risks that could invalidate the thesis.
           - time_horizon: "short-term" (days), "medium-term" (weeks), or
             "long-term" (months).
           - position_sizing_hint: "small", "medium", or "full" relative to
             the ticker's current portfolio weight.

        4. **Discuss reasoning, risks, and monitoring** — after the trade
           ideas, add a brief "monitoring" section listing events or data
           releases the user should watch for.

        OUTPUT FORMAT — respond with a single JSON object (no markdown fences,
        no commentary outside the JSON):

        {
          "per_ticker": {
            "<SYMBOL>": {
              "sentiment": "bullish" | "bearish" | "neutral",
              "key_news": ["<headline 1>", "<headline 2>", ...],
              "notes": "<brief analyst-style commentary>"
            },
            ...
          },
          "ideas": [
            {
              "ticker": "<SYMBOL>",
              "action": "buy" | "sell" | "hold",
              "thesis": "<concise thesis>",
              "risks": "<key risks>",
              "time_horizon": "short-term" | "medium-term" | "long-term",
              "position_sizing_hint": "small" | "medium" | "full"
            },
            ...  (at most 3)
          ],
          "monitoring": [
            "<event or data point to watch>",
            ...
          ],
          "disclaimer": "This analysis is informational only and does NOT constitute financial advice.  Always consult a licensed financial advisor before making investment decisions."
        }

        IMPORTANT:
        - Be conservative — when in doubt, prefer "hold" over "buy" or "sell".
        - Always include the disclaimer field verbatim.
        - If no compelling trade ideas exist, return an empty "ideas" array and
          explain why in a "monitoring" entry.
    """)

    user_content = textwrap.dedent(f"""\
        Here is the user's portfolio configuration:

        {portfolio_json}

        Here are the recent news articles:

        {news_json}

        Please analyse the news against the portfolio and produce your
        structured JSON response.
    """)

    return [
        Message(system_prompt, role="assistant"),
        Message(user_content, role="user"),
    ]


@mcp.prompt(
    name="daily_briefing",
    description=textwrap.dedent("""\
        A prompt template for generating a concise daily news briefing from the
        daily_news_snapshot resource.

        When rendered, this prompt instructs the LLM to:

          1. Summarise the most impactful macro-economic, geopolitical, and
             sector-specific developments from the provided news snapshot.
          2. Highlight which tickers, sectors, or asset classes are most
             affected.
          3. Output a structured response in either Markdown or JSON format
             suitable for a daily report email or dashboard widget.

        The agent typically reads the daily_news_snapshot resource, then
        passes its content to this prompt for summarisation.

        DISCLAIMER: The prompt explicitly instructs the model that its output
        is informational only and does NOT constitute financial advice.
    """),
)
def daily_briefing(
    snapshot_json: str,
    output_format: str = "markdown",
) -> list[Message]:
    """Build the daily-briefing prompt.

    Args:
        snapshot_json: The full daily news snapshot JSON string (as returned
            by the 'data://daily_news_snapshot' resource).  Must be valid JSON
            with an 'articles' array.
        output_format: Desired output format — either "markdown" (default)
            for a human-readable briefing, or "json" for a machine-parseable
            structured summary.  Defaults to "markdown".
    """
    if output_format not in ("markdown", "json"):
        output_format = "markdown"

    if output_format == "json":
        format_instructions = textwrap.dedent("""\
            OUTPUT FORMAT — respond with a single JSON object:
            {
              "date": "<snapshot date>",
              "executive_summary": "<2-3 sentence overview of the day>",
              "macro": {
                "headline": "<key macro development>",
                "details": "<1-2 sentence elaboration>",
                "affected_sectors": ["<sector>", ...]
              },
              "geopolitical": {
                "headline": "<key geopolitical development>",
                "details": "<1-2 sentence elaboration>",
                "affected_regions": ["<region>", ...]
              },
              "sector_highlights": [
                {
                  "sector": "<sector name>",
                  "development": "<what happened>",
                  "tickers_affected": ["<TICKER>", ...],
                  "sentiment": "positive" | "negative" | "neutral"
                },
                ...
              ],
              "watchlist": [
                "<event or data release to watch today>"
              ],
              "disclaimer": "This briefing is informational only and does NOT constitute financial advice."
            }
        """)
    else:
        format_instructions = textwrap.dedent("""\
            OUTPUT FORMAT — respond in clean Markdown with the following sections:

            # Daily Market Briefing — <date>

            ## Executive Summary
            <2-3 sentence overview of the day's most important developments>

            ## Macro-Economic Developments
            <bullet points summarising key macro news and their market implications>

            ## Geopolitical Highlights
            <bullet points summarising geopolitical developments>

            ## Sector & Ticker Spotlight
            <for each affected sector/ticker, a brief note on what happened and
            the likely directional impact>

            ## Watchlist & Upcoming Events
            <bullet points listing events, data releases, or earnings to monitor>

            ---
            *Disclaimer: This briefing is informational only and does NOT
            constitute financial advice.  Always consult a licensed financial
            advisor before making investment decisions.*
        """)

    system_prompt = textwrap.dedent(f"""\
        You are a senior macro-economic analyst preparing a **daily market
        briefing** for a portfolio manager.  You will be given a JSON snapshot
        of today's most important global news articles.

        Your task:
        1. Identify the 3-5 most impactful developments across macro-economics,
           geopolitics, and sector-specific news.
        2. For each development, explain its potential market impact in 1-2
           concise sentences.
        3. Highlight specific tickers, sectors, or asset classes that are most
           affected.
        4. End with a short watchlist of upcoming events or data releases.

        Be concise, analytical, and avoid speculation.  Stick to what the
        articles actually say.

        {format_instructions}
    """)

    user_content = textwrap.dedent(f"""\
        Here is today's news snapshot:

        {snapshot_json}

        Please produce the daily briefing.
    """)

    return [
        Message(system_prompt, role="assistant"),
        Message(user_content, role="user"),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info(
        "Starting News & Stocks MCP Server  "
        "DATA_DIR=%s  MAX_LIMIT=%d  HTTP_TIMEOUT=%.1fs",
        DATA_DIR,
        MAX_LIMIT,
        HTTP_TIMEOUT,
    )
    mcp.run()
