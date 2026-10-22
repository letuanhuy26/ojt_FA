# Data source and use status

- Symbol: SPY (single-asset US-listed ETF; provisional market choice).
- Source: Yahoo Finance historical-price endpoint, accessed using `yfinance`.
- Fields: daily OHLC, adjusted close, volume, dividends, and split actions.
- Retrieval: performed by `scripts/download_data.py`; request and local file
  hash are recorded in the generated `SPY_download_manifest.json`.
- Rights: unresolved for public redistribution or publication. The `yfinance`
  project states its client is intended for personal use and points users to
  Yahoo's terms. This copy is for local method development only; no raw price
  data are committed to the repository.
- Limitations: vendor corrections may change a later download; this is a
  single ETF and does not support a multi-market or constituent-level claim.

Before confirmation/final evaluation or external submission, either document
permission for the chosen source and use, or replace it with a licensed source
and rerun the locked protocol.
