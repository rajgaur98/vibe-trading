import os
import duckdb
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_path: str = None, read_only: bool = False):
        if not db_path:
            db_path = os.getenv("DATABASE_PATH", "data/vibe_trading.db")
        
        self.db_path = db_path
        self.read_only = read_only
        self.conn = None

    def connect(self):
        """Establishes connection to DuckDB."""
        if not self.read_only:
            # Ensure the directory exists
            parent_dir = Path(self.db_path).parent
            parent_dir.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Connecting to DuckDB at {self.db_path} (read_only={self.read_only})")
        self.conn = duckdb.connect(self.db_path, read_only=self.read_only)
        
        if not self.read_only:
            self._create_tables()

    def close(self):
        """Closes the connection."""
        if self.conn:
            self.conn.close()
            logger.info("DuckDB connection closed.")

    def _create_tables(self):
        """Initializes tables for candles, features, trade log, and decision log."""
        # 1. Candles table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                symbol VARCHAR,
                timeframe VARCHAR,
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                PRIMARY KEY (symbol, timeframe, timestamp)
            )
        """)

        # 2. Features table
        # We store features as a flexible schema. We can dynamically alter table or just store as JSON or pre-defined columns.
        # Storing pre-defined columns is cleaner for DuckDB queries.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS features (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                rsi_14 DOUBLE,
                rsi_regime VARCHAR,
                macd DOUBLE,
                macd_signal DOUBLE,
                macd_hist DOUBLE,
                macd_regime VARCHAR,
                adx_14 DOUBLE,
                adx_regime VARCHAR,
                obv DOUBLE,
                obv_trend VARCHAR,
                support_price DOUBLE,
                support_distance_pct DOUBLE,
                support_proximity VARCHAR,
                resistance_price DOUBLE,
                resistance_distance_pct DOUBLE,
                resistance_proximity VARCHAR,
                candlestick_pattern VARCHAR,
                funding_rate VARCHAR,
                open_interest_trend VARCHAR,
                is_macro_event_today BOOLEAN,
                PRIMARY KEY (symbol, timestamp)
            )
        """)

        # 3. Trades table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id VARCHAR PRIMARY KEY,
                symbol VARCHAR,
                action VARCHAR,
                entry_time TIMESTAMP,
                entry_price DOUBLE,
                close_time TIMESTAMP,
                close_price DOUBLE,
                size_usd DOUBLE,
                realized_pnl DOUBLE,
                result VARCHAR -- 'win' or 'loss'
            )
        """)

        # 4. Decision Log
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS decision_log (
                decision_id VARCHAR PRIMARY KEY,
                timestamp TIMESTAMP,
                symbol VARCHAR,
                action VARCHAR,
                stop_loss_strategy VARCHAR,
                take_profit_strategy VARCHAR,
                risk_reward_ratio DOUBLE,
                reasoning_summary VARCHAR,
                agent_transcripts VARCHAR -- JSON string of the transcripts
            )
        """)

        # 5. Portfolio State
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_state (
                timestamp TIMESTAMP PRIMARY KEY,
                balance DOUBLE,
                peak_balance DOUBLE
            )
        """)

        # 6. Open Positions
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS open_positions (
                symbol VARCHAR PRIMARY KEY,
                side VARCHAR,
                entry_time TIMESTAMP,
                entry_price DOUBLE,
                size_usd DOUBLE,
                stop_price DOUBLE,
                take_profit_price DOUBLE
            )
        """)

        logger.info("Database schemas verified.")
