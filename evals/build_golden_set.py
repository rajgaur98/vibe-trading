"""Generate real golden-set YAML cases from DuckDB-backed candles.

For each hand-picked (symbol, timestamp) below, this script:
  1. Runs `FeaturePipeline.run()` to produce the deterministic market snapshot.
  2. Derives AnalystLabel + TraderLabel field values from the snapshot's regime
     columns (RSI/MACD/ADX/OBV regimes + S/R proximity) using transparent voting
     rules. The LLM analyst still has to read the raw numbers and synthesize
     them; this script just encodes what a textbook Murphy-style interpretation
     produces given those same numbers.
  3. Picks a category-specific must-mention / must-not-mention rubric for the
     free-text fields (thesis, reasoning_summary).
  4. Writes one YAML per case to `evals/snapshots/`.

The (symbol, timestamp) list was curated via `evals/scan_candidates.py` to span
8 distinct regime buckets. This generator is committed so the derivation logic
is reviewable; rerun it after curation changes.

Usage:
    uv run python -m evals.build_golden_set
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from vibe_trading.data.db import Database
from vibe_trading.features.pipeline import FeaturePipeline


# Curated candidates — one per regime category, chosen from scan_candidates output.
# Each row: (symbol, ISO timestamp, category, short slug for filename).
CANDIDATES: list[tuple[str, str, str, str]] = [
    # Strong bullish trend with volume confirmation
    ("SOL/USDT",   "2026-02-07T00:00:00Z", "strong_bullish_trend",            "sol-strong-bullish"),
    ("LINK/USDT",  "2026-02-07T00:00:00Z", "strong_bullish_trend",            "link-strong-bullish"),
    ("PENGU/USDT", "2026-02-06T00:00:00Z", "strong_bullish_trend",            "pengu-strong-bullish"),

    # Bullish breakout at resistance
    ("BTC/USDT",   "2026-01-15T00:00:00Z", "bullish_breakout_at_resistance",  "btc-jan-breakout"),
    ("BTC/USDT",   "2026-04-05T00:00:00Z", "bullish_breakout_at_resistance",  "btc-apr-breakout"),

    # Strong bearish trend with volume confirmation
    ("ETH/USDT",   "2026-05-15T00:00:00Z", "strong_bearish_trend",            "eth-may-bearish"),
    ("SOL/USDT",   "2026-05-08T00:00:00Z", "strong_bearish_trend",            "sol-may-bearish"),

    # Overbought exhaustion
    ("BTC/USDT",   "2026-01-05T00:00:00Z", "overbought_exhaustion",           "btc-jan-overbought"),
    ("BTC/USDT",   "2026-05-05T00:00:00Z", "overbought_exhaustion",           "btc-may-overbought"),

    # Oversold bounce setup
    ("BTC/USDT",   "2026-01-25T00:00:00Z", "oversold_bounce_setup",           "btc-jan-oversold"),
    ("ETH/USDT",   "2026-01-25T00:00:00Z", "oversold_bounce_setup",           "eth-jan-oversold"),

    # Range-bound chop
    ("BTC/USDT",   "2026-04-25T00:00:00Z", "range_bound_chop",                "btc-apr-range"),
    ("ETH/USDT",   "2026-04-25T00:00:00Z", "range_bound_chop",                "eth-apr-range"),

    # Bullish price / OBV divergent (textbook fakeout setup)
    ("SUI/USDT",   "2026-01-28T00:00:00Z", "bullish_price_obv_divergent",     "sui-volume-divergence"),

    # --- Directional LONG cases (derivation yields action=long; exercise entry-strategy fields) ---
    ("ETH/USDT",   "2026-04-15T16:00:00Z", "bullish_directional",             "eth-apr-long"),
    ("SOL/USDT",   "2026-02-25T16:00:00Z", "bullish_directional",             "sol-feb-long"),
    ("SUI/USDT",   "2026-03-16T04:00:00Z", "bullish_directional",             "sui-mar-long"),
    ("BTC/USDT",   "2026-04-08T08:00:00Z", "bullish_directional",             "btc-apr-long"),

    # --- Directional SHORT cases (derivation yields action=short; diverse symbols) ---
    ("ETH/USDT",   "2026-02-10T12:00:00Z", "bearish_directional",             "eth-feb-short"),
    ("LINK/USDT",  "2026-03-26T12:00:00Z", "bearish_directional",             "link-mar-short"),
    ("NEAR/USDT",  "2026-03-26T08:00:00Z", "bearish_directional",             "near-mar-short"),
    ("AVAX/USDT",  "2026-03-23T00:00:00Z", "bearish_directional",             "avax-mar-short"),

    # ------------------------------------------------------------------
    # Expansion batch (023+): broaden symbol coverage to the deeper-history
    # alts (DOT / LTC / ONDO / TAO — previously absent) and add more cases in
    # under-represented regimes (range chop, oversold, overbought, volume
    # divergence, transitional). Curated from a denser `scan_candidates` pass
    # over these symbols; every (symbol, ts) below produces a non-empty
    # snapshot. Category tags reflect the snapshot's actual derived regime.
    # ------------------------------------------------------------------
    # New-symbol strong-bearish shorts (DOT / LTC / TAO never appeared before)
    ("DOT/USDT",   "2026-03-01T16:00:00Z", "strong_bearish_trend",            "dot-mar-bearish"),
    ("LTC/USDT",   "2026-03-18T08:00:00Z", "strong_bearish_trend",            "ltc-mar-bearish"),
    ("TAO/USDT",   "2026-03-18T00:00:00Z", "transitional",                    "tao-mar-transitional"),

    # New-symbol directional long (ONDO bullish breakout at resistance)
    ("ONDO/USDT",  "2026-05-23T20:00:00Z", "bullish_breakout_at_resistance",  "ondo-may-long"),

    # Overbought exhaustion on a new symbol (LTC) — bias bullish but flat (don't chase)
    ("LTC/USDT",   "2026-02-25T12:00:00Z", "overbought_exhaustion",           "ltc-feb-overbought"),

    # Oversold bounce setups on new symbols (DOT / ONDO)
    ("DOT/USDT",   "2026-01-19T00:00:00Z", "oversold_bounce_setup",           "dot-jan-oversold"),
    ("ONDO/USDT",  "2026-01-18T20:00:00Z", "oversold_bounce_setup",           "ondo-jan-oversold"),

    # Range-bound chop on a new symbol (LTC) — neutral, no edge
    ("LTC/USDT",   "2026-02-17T04:00:00Z", "range_bound_chop",                "ltc-feb-range"),

    # Volume-divergence / transitional spread across new + existing symbols
    ("DOT/USDT",   "2026-02-04T16:00:00Z", "bullish_price_obv_divergent",     "dot-feb-divergence"),
    ("TAO/USDT",   "2026-01-27T00:00:00Z", "bullish_breakout_at_resistance",  "tao-jan-breakout"),
    ("AVAX/USDT",  "2026-02-09T08:00:00Z", "transitional",                    "avax-feb-transitional"),
    ("NEAR/USDT",  "2026-01-22T20:00:00Z", "transitional",                    "near-jan-transitional"),
]


# ----------------------------------------------------------------------
# Snapshot → label derivation
# ----------------------------------------------------------------------

def _bias_votes(snapshot: dict) -> tuple[int, int]:
    """Count bullish vs bearish indicator votes from the snapshot's regime fields."""
    bull = 0
    bear = 0

    if snapshot["macd_hist"] > 0:
        bull += 1
    elif snapshot["macd_hist"] < 0:
        bear += 1

    if snapshot["rsi_14"] >= 55.0:
        bull += 1
    elif snapshot["rsi_14"] <= 45.0:
        bear += 1

    if snapshot["obv_trend"] == "accumulation":
        bull += 1
    elif snapshot["obv_trend"] == "distribution":
        bear += 1

    if snapshot["adx_regime"] == "strong_trend":
        # ADX is direction-agnostic; lean it the way MACD points
        if snapshot["macd_regime"].startswith("bullish"):
            bull += 1
        elif snapshot["macd_regime"].startswith("bearish"):
            bear += 1

    return bull, bear


def derive_market_bias(snapshot: dict) -> str:
    bull, bear = _bias_votes(snapshot)
    if bull - bear >= 2:
        return "bullish"
    if bear - bull >= 2:
        return "bearish"
    return "neutral"


def derive_volume_confirmation(snapshot: dict, market_bias: str) -> str:
    obv = snapshot["obv_trend"]
    if market_bias == "bullish":
        return "confirmed" if obv == "accumulation" else ("divergent" if obv == "distribution" else "weak")
    if market_bias == "bearish":
        return "confirmed" if obv == "distribution" else ("divergent" if obv == "accumulation" else "weak")
    return "weak"


def derive_confluence_score(snapshot: dict) -> float:
    bull, bear = _bias_votes(snapshot)
    return round(abs(bull - bear) / 4.0, 2)


def derive_trader_action(market_bias: str, confluence_score: float, snapshot: dict) -> str:
    """Map bias + confluence to an action, with Murphy-style overrides at RSI extremes.

    Even with a bullish bias, an overbought reading argues for waiting on a pullback
    rather than chasing — so we force `flat`. Symmetric override for oversold + bearish.
    """
    rsi_regime = snapshot["rsi_regime"]

    if rsi_regime == "overbought" and market_bias == "bullish":
        return "flat"  # don't chase the top
    if rsi_regime == "oversold" and market_bias == "bearish":
        return "flat"  # don't short into oversold

    if market_bias == "bullish" and confluence_score >= 0.6:
        return "long"
    if market_bias == "bearish" and confluence_score >= 0.6:
        return "short"
    return "flat"


# Rubric selection is driven by the DERIVED labels (not by the manual category
# tag) so that the must-mention / must-not-mention criteria always match what
# the snapshot actually shows. The category tag is informational only — used
# in the case description for traceability back to the regime intent.

def select_thesis_rubric(
    market_bias: str,
    volume_confirmation: str,
    snapshot: dict,
) -> dict[str, list[str]]:
    """Pick the thesis rubric that matches the derived bias + volume state.

    For overbought / oversold cases (extreme RSI), we override toward
    exhaustion / bounce framing regardless of overall bias since that's the
    load-bearing pattern.
    """
    rsi_regime = snapshot["rsi_regime"]

    if rsi_regime == "overbought":
        return {
            "must_mention":     ["overbought RSI", "momentum exhaustion risk"],
            "must_not_mention": ["chase the move higher"],
        }
    if rsi_regime == "oversold":
        return {
            "must_mention":     ["oversold RSI", "potential mean-reversion"],
            "must_not_mention": ["sell into the dip"],
        }

    if market_bias == "bullish":
        if volume_confirmation == "confirmed":
            return {
                "must_mention":     ["bullish bias", "volume confirms the move"],
                "must_not_mention": ["bearish reversal warning"],
            }
        if volume_confirmation == "divergent":
            return {
                "must_mention":     ["bullish price action", "volume divergence cautions"],
                "must_not_mention": ["strong volume-confirmed rally"],
            }
        return {
            "must_mention":     ["bullish bias", "weak volume confirmation"],
            "must_not_mention": ["strong-conviction breakout call"],
        }

    if market_bias == "bearish":
        if volume_confirmation == "confirmed":
            return {
                "must_mention":     ["bearish bias", "distribution confirms weakness"],
                "must_not_mention": ["bullish reversal call"],
            }
        if volume_confirmation == "divergent":
            return {
                "must_mention":     ["bearish price action", "accumulation divergence"],
                "must_not_mention": ["aggressive short conviction"],
            }
        return {
            "must_mention":     ["bearish bias", "weak volume confirmation"],
            "must_not_mention": ["high-conviction directional call"],
        }

    # neutral
    return {
        "must_mention":     ["lack of directional edge", "mixed indicator signals"],
        "must_not_mention": ["high-conviction directional call"],
    }


def select_reasoning_rubric(action: str) -> dict[str, list[str]]:
    if action == "long":
        return {
            "must_mention":     ["align with bullish bias", "asymmetric reward"],
            "must_not_mention": ["enter despite weak confluence"],
        }
    if action == "short":
        return {
            "must_mention":     ["align with bearish bias", "risk-managed entry"],
            "must_not_mention": ["fight the trend"],
        }
    # flat
    return {
        "must_mention":     ["lack of clear edge"],
        "must_not_mention": ["enter despite weak signal"],
    }


def build_case_dict(
    case_id: str,
    symbol: str,
    timestamp: datetime,
    snapshot: dict,
    category: str,
) -> dict:
    market_bias = derive_market_bias(snapshot)
    volume_confirmation = derive_volume_confirmation(snapshot, market_bias)
    confluence = derive_confluence_score(snapshot)
    action = derive_trader_action(market_bias, confluence, snapshot)

    # Trader strategy labels — encode the SAME house methodology the trader prompt
    # uses, and depend ONLY on signals the trader can also see (current price vs the
    # analyst's S/R levels => proximity). A level is "near" within 2%, which is exactly
    # the pipeline's very_close / immediate_contact proximity buckets.
    #   LONG  stop: swing_low if support near, else 1.5_atr; TP next_resistance.
    #   SHORT stop: tight_atr if resistance near, else 1.5_atr; TP 3.0_atr.
    #   RR target 2.0. hold_period_bias is not scored (kept 'medium' for completeness).
    if action == "long":
        stop_loss_strategy = "swing_low" if snapshot["support_proximity"] in {"very_close", "immediate_contact"} else "1.5_atr"
        take_profit_strategy = "next_resistance"
        risk_reward_ratio = 2.0
        hold_period_bias = "medium"
    elif action == "short":
        stop_loss_strategy = "tight_atr" if snapshot["resistance_proximity"] in {"very_close", "immediate_contact"} else "1.5_atr"
        take_profit_strategy = "3.0_atr"
        risk_reward_ratio = 2.0
        hold_period_bias = "medium"
    else:  # flat
        stop_loss_strategy = "1.5_atr"  # placeholder; not scored when flat
        take_profit_strategy = "risk_reward_multiplier"
        risk_reward_ratio = 1.5
        hold_period_bias = "medium"

    description = (
        f"{category.replace('_', ' ').title()} for {symbol} at {timestamp.isoformat()}. "
        f"Derived labels: bias={market_bias}, volume={volume_confirmation}, confluence={confluence}, "
        f"trader_action={action}. RSI={snapshot['rsi_14']:.1f} ({snapshot['rsi_regime']}), "
        f"MACD_hist={snapshot['macd_hist']:.4f} ({snapshot['macd_regime']}), "
        f"OBV={snapshot['obv_trend']}, ADX={snapshot['adx_14']:.1f} ({snapshot['adx_regime']})."
    )

    return {
        "id": case_id,
        "description": description,
        "symbol": symbol,
        "timestamp": timestamp,
        "analyst_label": {
            "market_bias": market_bias,
            "volume_confirmation": volume_confirmation,
            "nearest_support": round(float(snapshot["support_price"]), 4),
            "nearest_resistance": round(float(snapshot["resistance_price"]), 4),
            "confluence_score": confluence,
            "thesis_rubric": select_thesis_rubric(market_bias, volume_confirmation, snapshot),
        },
        "trader_label": {
            "action": action,
            "stop_loss_strategy": stop_loss_strategy,
            "take_profit_strategy": take_profit_strategy,
            "risk_reward_ratio": risk_reward_ratio,
            "hold_period_bias": hold_period_bias,
            "reasoning_rubric": select_reasoning_rubric(action),
        },
    }


def main() -> None:
    db = Database()
    db.connect()
    pipeline = FeaturePipeline(db)

    out_dir = Path("evals/snapshots")
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    for idx, (symbol, iso_ts, category, slug) in enumerate(CANDIDATES, start=1):
        timestamp = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        # FeaturePipeline manages its own DB connection per call
        db.close()
        try:
            snapshot = pipeline.run(symbol, timestamp)
        except Exception as e:
            print(f"  ✗ {symbol}@{iso_ts}: pipeline failed: {e}")
            db.connect()
            continue
        db.connect()

        if not snapshot:
            print(f"  ✗ {symbol}@{iso_ts}: empty snapshot — skipping")
            continue

        case_id = f"{idx:03d}-{slug}"
        case = build_case_dict(case_id, symbol, timestamp, snapshot, category)
        out_path = out_dir / f"{case_id}.yaml"
        out_path.write_text(yaml.safe_dump(case, sort_keys=False, default_flow_style=False))
        print(f"  ✓ {case_id}  ({symbol}@{iso_ts})  → {category} / bias={case['analyst_label']['market_bias']}, action={case['trader_label']['action']}")
        generated += 1

    db.close()
    print()
    print(f"Generated {generated} golden-set cases in {out_dir}/")


if __name__ == "__main__":
    main()
