# Data policy

Raw Yahoo Finance files and derived feature tables stay local and are excluded
from Git. The downloader records the request, library version, retrieval time,
row count, date range, column set, and SHA-256 digest in `data/manifests/`.

The present SPY file is a development pilot only. The `yfinance` project says
its Yahoo Finance client is intended for personal use and directs users to
Yahoo's terms for data rights. Do not redistribute the downloaded rows or
submit a paper based on them until a data source and publication rights have
been confirmed. See `data/manifests/source-and-rights.md`.
