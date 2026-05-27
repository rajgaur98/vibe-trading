"""One-shot scan: walk DuckDB candle history, bucket each (symbol, timestamp) by
market regime, and surface diverse candidates for the eval golden set.

Outputs nothing to disk — just prints a regime-categorized table to stdout so we
can pick which timestamps to commit as golden-set YAMLs.

Run: uv run python -m evals.scan_candidates
"""

from collections import defaultdict
from datetime import datetime

from vibe_trading.data.db import Database
from vibe_trading.features.pipeline import FeaturePipeline


# Major pairs with the deepest candle history
SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "SUI/USDT",
    "LINK/USDT", "AVAX/USDT", "DOT/USDT", "LTC/USDT",
    "NEAR/USDT", "ONDO/USDT", "TAO/USDT", "PENGU/USDT",
]

# Sample every 60 4h-candles (~10 days) — enough variety, manageable count
SAMPLE_STRIDE = 60


def categorize(snapshot: dict) -> str:
    """Classify a snapshot into one of ~8 regime buckets."""
    rsi = snapshot["rsi_14"]
    rsi_regime = snapshot["rsi_regime"]
    macd_hist = snapshot["macd_hist"]
    macd_regime = snapshot["macd_regime"]
    adx = snapshot["adx_14"]
    obv_trend = snapshot["obv_trend"]
    sup_prox = snapshot["support_proximity"]
    res_prox = snapshot["resistance_proximity"]

    bullish_momentum = macd_regime.startswith("bullish") and macd_hist > 0
    bearish_momentum = macd_regime.startswith("bearish") and macd_hist < 0
    strong_trend = adx >= 25.0
    weak_trend = adx < 20.0

    # 1. Overbought exhaustion: extreme RSI, momentum waning
    if rsi_regime == "overbought":
        return "overbought_exhaustion"

    # 2. Oversold bounce setup: extreme RSI, near support
    if rsi_regime == "oversold":
        return "oversold_bounce_setup"

    # 3. Strong bullish trend with volume confirmation
    if bullish_momentum and strong_trend and obv_trend == "accumulation":
        return "strong_bullish_trend"

    # 4. Strong bearish trend with volume confirmation
    if bearish_momentum and strong_trend and obv_trend == "distribution":
        return "strong_bearish_trend"

    # 5. Bullish breakout at resistance
    if bullish_momentum and res_prox in {"immediate_contact", "very_close"}:
        return "bullish_breakout_at_resistance"

    # 6. Volume divergence: price up but OBV distribution (or vice versa)
    if bullish_momentum and obv_trend == "distribution":
        return "bullish_price_obv_divergent"
    if bearish_momentum and obv_trend == "accumulation":
        return "bearish_price_obv_divergent"

    # 7. Range chop: weak trend, mid RSI
    if weak_trend and rsi_regime == "neutral":
        return "range_bound_chop"

    # 8. Default
    return "transitional"


def main():
    db = Database()
    db.connect()
    pipeline = FeaturePipeline(db)

    buckets: dict[str, list[tuple[str, datetime, dict]]] = defaultdict(list)

    for symbol in SYMBOLS:
        # Fetch all 4h timestamps for this symbol, then sample every SAMPLE_STRIDE
        ts_rows = db.conn.execute(
            "SELECT timestamp FROM candles WHERE symbol = ? AND timeframe = '4h' ORDER BY timestamp ASC",
            (symbol,),
        ).fetchall()
        if not ts_rows:
            continue
        # Skip the first 200 candles (indicator warm-up window) and sample
        sampled = ts_rows[200::SAMPLE_STRIDE]

        for (ts,) in sampled:
            db.close()
            try:
                snapshot = pipeline.run(symbol, ts)
            except Exception:
                snapshot = None
            db.connect()

            if not snapshot:
                continue
            cat = categorize(snapshot)
            buckets[cat].append((symbol, ts, snapshot))

    db.close()

    # Print summary
    print(f"{'CATEGORY':<35s} {'COUNT':>6s}  EXAMPLES")
    print("─" * 90)
    for cat in sorted(buckets.keys()):
        items = buckets[cat]
        # Show 3 examples per bucket
        examples = ", ".join(f"{s}@{ts.strftime('%Y-%m-%d')}" for s, ts, _ in items[:3])
        print(f"{cat:<35s} {len(items):>6d}  {examples}")


if __name__ == "__main__":
    main()
