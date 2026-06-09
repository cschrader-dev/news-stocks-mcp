# News & Stocks MCP Server

A production-grade **Model Context Protocol (MCP)** server that provides AI agents with structured access to global news and finance/stock news from free, publicly accessible RSS sources.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    MCP Host / Agent                   │
│         (Claude Desktop, custom client, etc.)         │
└────────────────────────┬─────────────────────────────┘
                         │ stdio / SSE
┌────────────────────────┴─────────────────────────────┐
│              News & Stocks MCP Server                 │
│                                                       │
│  TOOLS                                                │
│    • get_global_news(query, limit)                    │
│    • get_stock_news(symbol, limit)                    │
│    • get_market_snapshot(symbols, limit_per_symbol)   │
│                                                       │
│  RESOURCES                                            │
│    • data://portfolio          → portfolio.json       │
│    • data://daily_news_snapshot → snapshot.json        │
│                                                       │
│  PROMPTS                                              │
│    • portfolio_news_recommendations                   │
│    • daily_briefing                                   │
└──────────────────────────────────────────────────────┘
         │                              │
    Google News RSS              Local JSON files
    (free, no API key)           (data/ directory)
```

## Quick Start

```bash
# 1. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server (stdio transport)
python news_mcp_server.py
```

## Configuration

All configuration is via environment variables with safe defaults:

| Variable | Default | Description |
|---|---|---|
| `NEWS_SEARCH_URL` | Google News RSS search | URL template for global news (`{query}` placeholder) |
| `STOCK_NEWS_URL` | Google News RSS search | URL template for stock news (`{symbol}` placeholder) |
| `HTTP_TIMEOUT` | `15.0` | HTTP request timeout in seconds |
| `MAX_LIMIT` | `50` | Upper bound on articles per request |
| `DATA_DIR` | `./data` | Directory containing portfolio and snapshot JSON files |
| `LOG_LEVEL` | `INFO` | Python logging level |

## MCP Primitives

### Tools

| Tool | Description |
|---|---|
| `get_global_news(query, limit)` | Fetch global news articles for any search query |
| `get_stock_news(symbol, limit)` | Fetch stock-specific news for a ticker symbol |
| `get_market_snapshot(symbols, limit_per_symbol)` | Batch-fetch news for multiple tickers |

### Resources

| URI | Description |
|---|---|
| `data://portfolio` | User's portfolio config (tickers, weights, risk prefs) |
| `data://daily_news_snapshot` | Cached daily global news snapshot |

### Prompts

| Prompt | Description |
|---|---|
| `portfolio_news_recommendations` | Trade-idea generation from portfolio + news |
| `daily_briefing` | Daily macro/sector briefing from news snapshot |

## Project Structure

```
MCP Server/
├── news_mcp_server.py          # Main MCP server (single-file)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── data/
    ├── portfolio.json          # Portfolio configuration
    └── daily_news_snapshot.json # Cached daily news snapshot
```

## Safety & Limitations

- **Read-only**: No file writes, no shell execution, no trades.
- **Outbound HTTP only**: Fetches RSS feeds from news aggregators.
- **No API keys**: Uses only free, public RSS feeds.
- **No financial advice**: All prompts include explicit disclaimers.

## License

MIT
