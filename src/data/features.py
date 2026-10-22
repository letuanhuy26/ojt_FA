from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "return_1",
    "intraday_return",
    "overnight_gap",
    "range_fraction",
    "volume_log_change",
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "volatility_5",
    "volatility_20",
    "volatility_60",
]
REGIME_COLUMNS = ["return_1", "volatility_20", "volume_log_change", "momentum_20"]


def build_examples(raw: pd.DataFrame) -> pd.DataFrame:
    """Construct date-t inputs and next-session open-to-close labels.

    OHLC is put on the adjusted-close basis using the vendor's adjustment
    factor. Every feature is computed from rows up through date t. The label
    and `target_date` come from the next observed trading session.
    """
    frame = raw.copy()
    if "Date" not in frame:
        frame = frame.rename_axis("Date").reset_index()
    frame["Date"] = pd.to_datetime(frame["Date"], utc=True).dt.tz_convert(None).dt.normalize()
    frame = frame.sort_values("Date").drop_duplicates("Date", keep=False).reset_index(drop=True)
    required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Raw file lacks required columns: {missing}")

    close = pd.to_numeric(frame["Close"], errors="coerce")
    adj_close = pd.to_numeric(frame["Adj Close"], errors="coerce")
    factor = adj_close / close
    open_adj = pd.to_numeric(frame["Open"], errors="coerce") * factor
    high_adj = pd.to_numeric(frame["High"], errors="coerce") * factor
    low_adj = pd.to_numeric(frame["Low"], errors="coerce") * factor
    volume = pd.to_numeric(frame["Volume"], errors="coerce")

    log_close = np.log(adj_close.where(adj_close > 0))
    log_open = np.log(open_adj.where(open_adj > 0))
    log_volume = np.log1p(volume.clip(lower=0))
    returns = log_close.diff()

    x = pd.DataFrame(index=frame.index)
    x["return_1"] = returns
    x["intraday_return"] = log_close - log_open
    x["overnight_gap"] = log_open - log_close.shift(1)
    x["range_fraction"] = (high_adj - low_adj) / adj_close.shift(1)
    x["volume_log_change"] = log_volume.diff()
    for lag in (5, 20, 60):
        x[f"momentum_{lag}"] = log_close - log_close.shift(lag)
        x[f"volatility_{lag}"] = returns.rolling(lag, min_periods=lag).std(ddof=0)

    next_intraday_return = x["intraday_return"].shift(-1)
    target_date = frame["Date"].shift(-1)
    output = x[FEATURE_COLUMNS].copy()
    output.insert(0, "date", frame["Date"])
    output["target_date"] = target_date
    output["target_return"] = next_intraday_return
    # A zero return belongs to the non-positive class by a fixed rule.
    output["target"] = (next_intraday_return > 0).astype("int8")
    output.loc[next_intraday_return.isna(), "target"] = np.nan
    output = output.dropna(subset=FEATURE_COLUMNS + ["target_date", "target_return", "target"])
    output["target"] = output["target"].astype("int8")
    output["target_date"] = pd.to_datetime(output["target_date"])
    return output.reset_index(drop=True)
