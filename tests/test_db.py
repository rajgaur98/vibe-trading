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
