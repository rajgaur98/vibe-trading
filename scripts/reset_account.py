"""One-off: reset the paper trading account to a clean slate.

Backs up the four account-state tables to a timestamped JSON file, then clears
them in Supabase Postgres (the authoritative store the live broker uses) and
seeds a fresh portfolio_state row at the starting balance. Market data
(candles, features) is left untouched. DuckDB's vestigial copies are cleared
best-effort.

Run: uv run python scripts/reset_account.py
"""

import os
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from vibe_trading.data.db import Database, PostgresDatabase

ACCOUNT_TABLES = ["open_positions", "trades", "decision_log", "portfolio_state"]
STARTING_BALANCE = 10000.0


def _seed_balance() -> float:
    """The portfolio_state seed. In LIVE_TESTNET the account is the real Binance demo
    futures account — seed its ACTUAL balance so the dashboard's balance/peak/drawdown
    reflect reality, not a fictitious paper $10k. Falls back to STARTING_BALANCE."""
    if os.getenv("TRADING_MODE", "PAPER").upper() == "LIVE_TESTNET":
        try:
            from vibe_trading.brokers.binance_futures import BinanceFuturesBroker
            bal = float(BinanceFuturesBroker(db=None).get_balance())
            print(f"LIVE_TESTNET: seeding portfolio_state from real demo balance ${bal:,.2f}")
            return bal
        except Exception as e:
            print(f"  (could not read live demo balance; seeding ${STARTING_BALANCE:,.2f}: {e})")
    return STARTING_BALANCE

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
backup_dir = Path("data/backups")
backup_dir.mkdir(parents=True, exist_ok=True)
backup_path = backup_dir / f"account-backup-{stamp}.json"

# ---- 1. Backup + clear Postgres (authoritative) ----------------------------
pg = PostgresDatabase()
pg.connect()
backup = {}
try:
    for tbl in ACCOUNT_TABLES:
        cur = pg.conn.execute(f"SELECT * FROM {tbl}")
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        backup[tbl] = rows
    backup_path.write_text(json.dumps(backup, indent=2, default=str))
    print(f"Backup written: {backup_path}")
    for tbl in ACCOUNT_TABLES:
        print(f"  backed up {len(backup[tbl]):>4d} rows from {tbl}")

    print("\nClearing Postgres account tables...")
    for tbl in ACCOUNT_TABLES:
        pg.conn.execute(f"DELETE FROM {tbl}")
    # Seed a fresh starting balance so the dashboard reflects the reset immediately.
    # In LIVE_TESTNET this is the real demo balance (peak = balance), not a paper $10k.
    seed_balance = _seed_balance()
    pg.conn.execute(
        "INSERT INTO portfolio_state (timestamp, balance, peak_balance) VALUES (CURRENT_TIMESTAMP, %s, %s)",
        (seed_balance, seed_balance),
    )
    pg.conn.commit()

    print("Verifying Postgres row counts after reset:")
    for tbl in ACCOUNT_TABLES:
        n = pg.conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl:<16s} {n} rows")
finally:
    pg.close()

# ---- 2. Best-effort clear of DuckDB vestigial copies (keep candles/features) ----
print("\nClearing DuckDB vestigial state tables (candles/features preserved)...")
duck = Database()
try:
    duck.connect()
    for tbl in ACCOUNT_TABLES:
        try:
            duck.conn.execute(f"DELETE FROM {tbl}")
        except Exception as e:
            print(f"  (skip {tbl}: {e})")
    print("  DuckDB state tables cleared.")
except Exception as e:
    print(f"  DuckDB cleanup skipped (non-fatal, state is vestigial): {e}")
finally:
    duck.close()

print(f"\nDone. Account reset to ${seed_balance:,.2f}. Backup at {backup_path}")
