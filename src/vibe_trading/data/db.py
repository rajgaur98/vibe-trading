import os
import time
import duckdb
from pathlib import Path
import logging
import psycopg2
from psycopg2.pool import ThreadedConnectionPool

logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_path: str = None, read_only: bool = False):
        if not db_path:
            db_path = os.getenv("DATABASE_PATH", "data/vibe_trading.db")
        
        self.db_path = db_path
        self.read_only = read_only
        self.conn = None

    def connect(self, retries: int = 5, backoff: float = 0.5):
        """Establishes connection to DuckDB with retry logic for lock contention."""
        if not self.read_only:
            # Ensure the directory exists
            parent_dir = Path(self.db_path).parent
            parent_dir.mkdir(parents=True, exist_ok=True)
        
        last_err = None
        for attempt in range(retries):
            try:
                logger.info(f"Connecting to DuckDB at {self.db_path} (read_only={self.read_only})")
                self.conn = duckdb.connect(self.db_path, read_only=self.read_only)
                
                if not self.read_only:
                    self._create_tables()
                return  # success
            except duckdb.IOException as e:
                last_err = e
                wait = backoff * (2 ** attempt)
                logger.warning(f"DuckDB lock contention (attempt {attempt + 1}/{retries}), retrying in {wait:.1f}s: {e}")
                time.sleep(wait)
        
        raise last_err  # all retries exhausted

    def close(self):
        """Closes the connection."""
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
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
                result VARCHAR, -- 'win' or 'loss'
                decision_id VARCHAR -- FK to decision_log.decision_id (links outcome to decision)
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
                agent_transcripts VARCHAR, -- JSON string of the agent reasoning transcripts
                trace_id VARCHAR -- Langfuse trace id (join a decision to its trace)
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
                take_profit_price DOUBLE,
                decision_id VARCHAR -- the decision that opened this position (carried to the closed trade)
            )
        """)

        # Idempotent column migrations for pre-existing tables (CREATE IF NOT EXISTS won't
        # add columns to a table that already exists).
        for stmt in (
            "ALTER TABLE trades ADD COLUMN IF NOT EXISTS decision_id VARCHAR",
            "ALTER TABLE decision_log ADD COLUMN IF NOT EXISTS trace_id VARCHAR",
            "ALTER TABLE open_positions ADD COLUMN IF NOT EXISTS decision_id VARCHAR",
        ):
            try:
                self.conn.execute(stmt)
            except Exception:
                pass

        logger.info("Database schemas verified.")


def translate_query(sql: str) -> str:
    """Translates DuckDB SQL dialect to PostgreSQL dialect."""
    # Replace DuckDB placeholder '?' with PostgreSQL placeholder '%s'
    sql = sql.replace('?', '%s')

    # Translate dialect-specific commands
    if "INSERT OR IGNORE INTO decision_log" in sql:
        sql = sql.replace("INSERT OR IGNORE INTO decision_log", "INSERT INTO decision_log")
        sql += " ON CONFLICT (decision_id) DO NOTHING"
    elif "INSERT OR IGNORE INTO llm_cost_log" in sql:
        sql = sql.replace("INSERT OR IGNORE INTO llm_cost_log", "INSERT INTO llm_cost_log")
        sql += " ON CONFLICT (call_id) DO NOTHING"
    elif "INSERT OR REPLACE INTO open_positions" in sql:
        sql = sql.replace("INSERT OR REPLACE INTO open_positions", "INSERT INTO open_positions")
        sql += """ ON CONFLICT (symbol) DO UPDATE SET
            side = EXCLUDED.side,
            entry_time = EXCLUDED.entry_time,
            entry_price = EXCLUDED.entry_price,
            size_usd = EXCLUDED.size_usd,
            stop_price = EXCLUDED.stop_price,
            take_profit_price = EXCLUDED.take_profit_price,
            decision_id = EXCLUDED.decision_id"""
    return sql


class PostgresConnectionWrapper:
    """Wraps a psycopg2 connection to mimic DuckDB execution syntax."""
    def __init__(self, conn):
        self._conn = conn
        self._cur = None

    @property
    def connection(self):
        return self._conn

    def execute(self, sql: str, params=None):
        if not self._cur:
            self._cur = self._conn.cursor()
        translated_sql = translate_query(sql)
        self._cur.execute(translated_sql, params)
        return self._cur

    def fetchone(self):
        if self._cur:
            return self._cur.fetchone()
        return None

    def fetchall(self):
        if self._cur:
            return self._cur.fetchall()
        return []

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        if self._cur:
            try:
                self._cur.close()
            except Exception:
                pass
            self._cur = None


class PostgresDatabase:
    """Manages thread-safe connection pool to Supabase Postgres."""
    _pool = None

    def __init__(self, db_url: str = None):
        if not db_url:
            db_url = os.getenv("POSTGRES_URL")
        if not db_url:
            raise ValueError("POSTGRES_URL environment variable is not set. Please check your .env file.")
        
        self.db_url = db_url
        self.conn = None
        self._initialize_pool()
        self._create_tables()

    def _initialize_pool(self):
        """Initializes a shared ThreadedConnectionPool."""
        if PostgresDatabase._pool is None:
            try:
                logger.info("Initializing ThreadedConnectionPool to Supabase Postgres...")
                # Min 1, Max 15 connections
                PostgresDatabase._pool = ThreadedConnectionPool(1, 15, self.db_url)
            except Exception as e:
                logger.error(f"Failed to initialize Postgres connection pool: {e}")
                raise

    def connect(self):
        """Acquires a connection from the pool and wraps it."""
        if self.conn is None:
            try:
                raw_conn = PostgresDatabase._pool.getconn()
                self.conn = PostgresConnectionWrapper(raw_conn)
                logger.info("Acquired connection from Postgres pool.")
            except Exception as e:
                logger.error(f"Failed to get connection from pool: {e}")
                raise

    def close(self):
        """Returns the connection back to the pool."""
        if self.conn:
            try:
                # Commit any uncommitted transactions before returning
                self.conn.commit()
            except Exception:
                try:
                    self.conn.rollback()
                except Exception:
                    pass
            
            try:
                self.conn.close()
                PostgresDatabase._pool.putconn(self.conn.connection)
                logger.info("Returned connection to Postgres pool.")
            except Exception as e:
                logger.error(f"Error returning connection to pool: {e}")
            finally:
                self.conn = None

    def _create_tables(self):
        """Creates the relational/state tables if they do not exist on Supabase."""
        self.connect()
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_state (
                    timestamp TIMESTAMP PRIMARY KEY,
                    balance DOUBLE PRECISION,
                    peak_balance DOUBLE PRECISION
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS open_positions (
                    symbol VARCHAR PRIMARY KEY,
                    side VARCHAR,
                    entry_time TIMESTAMP,
                    entry_price DOUBLE PRECISION,
                    size_usd DOUBLE PRECISION,
                    stop_price DOUBLE PRECISION,
                    take_profit_price DOUBLE PRECISION,
                    decision_id VARCHAR
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id VARCHAR PRIMARY KEY,
                    symbol VARCHAR,
                    action VARCHAR,
                    entry_time TIMESTAMP,
                    entry_price DOUBLE PRECISION,
                    close_time TIMESTAMP,
                    close_price DOUBLE PRECISION,
                    size_usd DOUBLE PRECISION,
                    realized_pnl DOUBLE PRECISION,
                    result VARCHAR,
                    decision_id VARCHAR
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS decision_log (
                    decision_id VARCHAR PRIMARY KEY,
                    timestamp TIMESTAMP,
                    symbol VARCHAR,
                    action VARCHAR,
                    stop_loss_strategy VARCHAR,
                    take_profit_strategy VARCHAR,
                    risk_reward_ratio DOUBLE PRECISION,
                    reasoning_summary TEXT,
                    agent_transcripts TEXT,
                    trace_id VARCHAR
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_cost_log (
                    call_id VARCHAR PRIMARY KEY,
                    timestamp TIMESTAMP,
                    provider VARCHAR,
                    model VARCHAR,
                    call_type VARCHAR,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    cost_usd DOUBLE PRECISION,
                    latency_ms DOUBLE PRECISION,
                    cache_read_tokens INTEGER,
                    cache_write_tokens INTEGER,
                    schema_ok BOOLEAN
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS decision_embeddings (
                    decision_id VARCHAR PRIMARY KEY,
                    symbol VARCHAR,
                    timestamp TIMESTAMP,
                    action VARCHAR,
                    entry_price DOUBLE PRECISION,
                    setup_text TEXT,
                    embedding DOUBLE PRECISION[]
                )
            """)
            # Idempotent column migrations for pre-existing Supabase tables.
            for stmt in (
                "ALTER TABLE trades ADD COLUMN IF NOT EXISTS decision_id VARCHAR",
                "ALTER TABLE decision_log ADD COLUMN IF NOT EXISTS trace_id VARCHAR",
                "ALTER TABLE open_positions ADD COLUMN IF NOT EXISTS decision_id VARCHAR",
                "ALTER TABLE llm_cost_log ADD COLUMN IF NOT EXISTS cache_read_tokens INTEGER",
                "ALTER TABLE llm_cost_log ADD COLUMN IF NOT EXISTS cache_write_tokens INTEGER",
                "ALTER TABLE llm_cost_log ADD COLUMN IF NOT EXISTS schema_ok BOOLEAN",
            ):
                self.conn.execute(stmt)
            self.conn.commit()
            logger.info("Supabase Postgres tables verified successfully.")
        except Exception as e:
            logger.error(f"Failed to verify/create Supabase Postgres tables: {e}")
            try:
                self.conn.rollback()
            except Exception:
                pass
            raise
        finally:
            self.close()

