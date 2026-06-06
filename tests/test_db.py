import pytest
import os
from datetime import datetime
from vibe_trading.data.db import Database

def test_database_creation_and_candles_read_write(tmp_path):
    # Setup temporary DuckDB path
    db_file = tmp_path / "test_vibe.db"
    db = Database(db_path=str(db_file))
    
    db.connect()
    
    # Verify tables created
    tables_res = db.conn.execute("SHOW TABLES").fetchall()
    tables = [r[0] for r in tables_res]
    assert "candles" in tables
    assert "features" in tables
    assert "trades" in tables
    assert "decision_log" in tables
    
    # Test inserting candles
    now = datetime.utcnow()
    db.conn.execute("""
        INSERT INTO candles (symbol, timeframe, timestamp, open, high, low, close, volume)
        VALUES ('BTC/USDT', '4h', ?, 50000.0, 51000.0, 49000.0, 50500.0, 100.0)
    """, (now,))
    
    # Verify select works
    res = db.conn.execute("SELECT close FROM candles WHERE symbol = 'BTC/USDT'").fetchone()
    assert res[0] == 50500.0
    
    db.close()


def test_postgres_database():
    from dotenv import load_dotenv
    load_dotenv()
    # Only run if POSTGRES_URL is set in environment
    db_url = os.getenv("POSTGRES_URL")
    if not db_url:
        pytest.skip("POSTGRES_URL not set in environment. Skipping Postgres integration test.")

    from vibe_trading.data.db import PostgresDatabase
    db = PostgresDatabase(db_url=db_url)
    db.connect()

    # Clean up test symbol if present
    db.conn.execute("DELETE FROM open_positions WHERE symbol = ?", ("TEST/USDT",))
    db.conn.commit()

    # Test INSERT using the wrapper
    db.conn.execute("""
        INSERT OR REPLACE INTO open_positions (symbol, side, entry_time, entry_price, size_usd, stop_price, take_profit_price)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, ("TEST/USDT", "long", datetime.utcnow(), 10.0, 100.0, 9.0, 12.0))
    db.conn.commit()

    # Test SELECT
    res = db.conn.execute("SELECT side, entry_price FROM open_positions WHERE symbol = ?", ("TEST/USDT",)).fetchone()
    assert res is not None
    assert res[0] == "long"
    assert res[1] == 10.0

    # Clean up
    db.conn.execute("DELETE FROM open_positions WHERE symbol = ?", ("TEST/USDT",))
    db.conn.commit()
    db.close()



from vibe_trading.data.db import translate_query


def test_translate_query_handles_llm_cost_log_insert():
    sql = "INSERT OR IGNORE INTO llm_cost_log (call_id, cost_usd) VALUES (?, ?)"
    out = translate_query(sql)
    assert "INSERT INTO llm_cost_log" in out
    assert "ON CONFLICT (call_id) DO NOTHING" in out
    assert "?" not in out  # placeholders translated to %s
    assert "%s" in out
