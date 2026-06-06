"""Manual live smoke test for BinanceFuturesBroker against the Binance futures testnet.

NOT run by pytest. Requires BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET in your
environment (or .env). Places a tiny BTC long with a bracket, prints the resulting position
and open orders, then closes it.

Usage:
    python scripts/binance_testnet_smoke.py
"""
import logging
import time

from dotenv import load_dotenv

from vibe_trading.brokers.binance_futures import BinanceFuturesBroker

logging.basicConfig(level=logging.INFO)
load_dotenv()


def main():
    broker = BinanceFuturesBroker(db=None)  # real testnet via ccxt set_sandbox_mode

    symbol = "BTC/USDT"
    mark = broker.get_mark_price(symbol)
    print(f"Mark price for {symbol}: {mark}")
    if not mark:
        raise SystemExit("Could not read mark price — check creds / connectivity.")

    # ~$200 notional; TP +2%, SL -2%
    res = broker.submit_order(
        symbol=symbol, action="long", size_usd=200.0,
        stop_price=mark * 0.98, take_profit_price=mark * 1.02, entry_price=mark,
    )
    print(f"submit_order → {res}")
    if res["status"] not in ("success", "dry_run"):
        raise SystemExit(f"Order not placed: {res}")

    time.sleep(2)
    print("Open positions on exchange:")
    for p in broker.get_open_positions():
        print(f"  {p}")

    print("Closing position...")
    print(f"close_position → {broker.close_position(symbol)}")


if __name__ == "__main__":
    main()
