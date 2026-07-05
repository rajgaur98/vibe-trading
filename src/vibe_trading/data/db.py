import os
import time
import duckdb
from pathlib import Path
import logging
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
import numpy as np

logger = logging.getLogger(__name__)

# Dimensionality of journal embeddings (gemini/gemini-embedding-001 = 3072).
# Drives the pgvector column typmod, query casts, and index DDL.
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "3072"))


def resolve_embedding_dim(existing_len, env_dim: int) -> int:
    """The dim the migration must use: existing rows win (migrating a live table
    to the wrong typmod would corrupt it); env is the cold-start default."""
    if existing_len:
        return int(existing_len)
    if env_dim <= 0:
        raise ValueError(f"EMBEDDING_DIM must be positive, got {env_dim}")
    return env_dim


def pgvector_migration_statements(dim: int) -> list:
    """DDL to move decision_embeddings.embedding from float8[] to vector(dim) and
    index it. The index is pgvector's documented halfvec EXPRESSION form because
    HNSW on plain vector caps at 2000 dims (ours is 3072). Queries must use the
    byte-identical cast expression to hit the index (see journal.pgvector_topk_sql)."""
    return [
        f"ALTER TABLE decision_embeddings "
        f"ALTER COLUMN embedding TYPE vector({dim}) USING embedding::vector({dim})",
        f"CREATE INDEX IF NOT EXISTS decision_embeddings_embedding_hnsw "
        f"ON decision_embeddings "
        f"USING hnsw ((embedding::halfvec({dim})) halfvec_cosine_ops)",
    ]


def adapt_embedding(vec):
    """Adapt a Python list embedding for the active decision_embeddings column type:
    float32 ndarray when pgvector is registered (the pgvector psycopg2 adapter
    serializes ndarrays to vector literals), plain list for the float8[] fallback."""
    if PostgresDatabase.pgvector_enabled:
        return np.asarray(vec, dtype=np.float32)
    return list(vec)

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
                trace_id VARCHAR, -- Langfuse trace id (join a decision to its trace)
                prompt_version VARCHAR, -- prompts.bundle_version() at decision time
                precedents_k INTEGER -- how many journal precedents the trader saw
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
            "ALTER TABLE decision_log ADD COLUMN IF NOT EXISTS prompt_version VARCHAR",
            "ALTER TABLE decision_log ADD COLUMN IF NOT EXISTS precedents_k INTEGER",
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
    elif "INSERT OR IGNORE INTO decision_scores" in sql:
        sql = sql.replace("INSERT OR IGNORE INTO decision_scores",
                          "INSERT INTO decision_scores")
        sql += " ON CONFLICT (decision_id) DO NOTHING"
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
    pgvector_enabled = False   # extension present + column migrated to vector
    pgvector_halfvec = False   # halfvec type available (pgvector >= 0.7) -> HNSW index built

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
                if PostgresDatabase.pgvector_enabled:
                    try:
                        from pgvector.psycopg2 import register_vector
                        register_vector(raw_conn)  # idempotent per connection
                    except Exception as e:
                        logger.warning(f"pgvector register_vector failed (non-fatal): {e}")
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
                    trace_id VARCHAR,
                    prompt_version VARCHAR, -- prompts.bundle_version() at decision time
                    precedents_k INTEGER -- how many journal precedents the trader saw
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
                    schema_ok BOOLEAN,
                    prompt_version VARCHAR
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
            # Online-eval scores: one row per scored decision (see eval/online.py).
            # outcome_* is deterministic (PnL / counterfactual forward return);
            # judge_* is the sampled generic-rubric LLM judge, filled in later.
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS decision_scores (
                    decision_id VARCHAR PRIMARY KEY,
                    scored_at TIMESTAMP,
                    kind VARCHAR,                    -- 'closed' | 'counterfactual'
                    outcome_pct DOUBLE PRECISION,    -- signed % in the decision's favor (raw move for flat)
                    outcome_score DOUBLE PRECISION,  -- [0,1]
                    judge_score DOUBLE PRECISION,    -- [0,1], NULL until sampled
                    judge_note TEXT,
                    judged_at TIMESTAMP,
                    prompt_version VARCHAR,          -- copied from decision_log at scoring time
                    precedents_k INTEGER             -- how many journal precedents the trader saw
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
                "ALTER TABLE llm_cost_log ADD COLUMN IF NOT EXISTS prompt_version VARCHAR",
                "ALTER TABLE decision_log ADD COLUMN IF NOT EXISTS prompt_version VARCHAR",
                "ALTER TABLE decision_log ADD COLUMN IF NOT EXISTS precedents_k INTEGER",
                "ALTER TABLE decision_scores ADD COLUMN IF NOT EXISTS precedents_k INTEGER",
            ):
                self.conn.execute(stmt)

            self._enable_pgvector()

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

    def _get_embedding_column_udt(self):
        """Re-queries decision_embeddings.embedding's underlying type. Returns
        'vector' once migrated, '_float8' (or None if the table/column is
        somehow missing) otherwise. Always re-read from the catalog rather than
        assumed, so pgvector_enabled reflects reality even if a migration
        attempt partially failed."""
        row = self.conn.execute(
            "SELECT udt_name FROM information_schema.columns "
            "WHERE table_name = 'decision_embeddings' "
            "AND column_name = 'embedding'").fetchone()
        return row[0] if row else None

    def _enable_pgvector(self):
        """Fail-soft pgvector capability probe + one-time column migration.
        See docs/superpowers/plans/2026-07-05-retrieval-upgrade.md (Task D1)
        for the design. This method must NEVER raise — no code path may
        REQUIRE pgvector; on any unexpected failure we log a warning and the
        in-Python float8[] fallback (adapt_embedding / cosine_topk) is used.

        The column ALTER and the HNSW index CREATE are deliberately split into
        separate transactions: the column migration is committed immediately
        on success, so a subsequent index-build failure (e.g. pgvector < 0.7
        has no `halfvec` type) only rolls back the index attempt, not the
        already-successful column migration.
        """
        try:
            try:
                self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                self.conn.commit()
            except Exception:
                self.conn.rollback()  # no privilege / no extension -> fallback path

            ext = self.conn.execute(
                "SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
            if not ext:
                logger.info("pgvector extension not present — using in-Python retrieval.")
                return

            col_udt = self._get_embedding_column_udt()
            if col_udt == "_float8":  # still float8[] -> migrate once
                row = self.conn.execute(
                    "SELECT array_length(embedding, 1) FROM decision_embeddings "
                    "LIMIT 1").fetchone()
                dim = resolve_embedding_dim(row[0] if row else None, EMBEDDING_DIM)
                alter_stmt, index_stmt = pgvector_migration_statements(dim)
                try:
                    self.conn.execute(alter_stmt)
                    self.conn.commit()  # lock in the column migration independently
                    col_udt = self._get_embedding_column_udt()
                except Exception as e:
                    logger.warning(f"pgvector column migration failed: {e}")
                    self.conn.rollback()
                    col_udt = self._get_embedding_column_udt()

                if col_udt == "vector":
                    try:
                        self.conn.execute(index_stmt)
                        self.conn.commit()
                    except Exception as e:
                        # HNSW build may fail on pgvector < 0.7 (no halfvec type);
                        # the already-committed column migration is unaffected.
                        logger.warning(f"pgvector HNSW index build skipped: {e}")
                        self.conn.rollback()

            PostgresDatabase.pgvector_enabled = (col_udt == "vector")
            half = self.conn.execute(
                "SELECT 1 FROM pg_type WHERE typname = 'halfvec'").fetchone()
            PostgresDatabase.pgvector_halfvec = bool(half)
            logger.info(
                f"pgvector enabled={PostgresDatabase.pgvector_enabled} "
                f"halfvec={PostgresDatabase.pgvector_halfvec}."
            )
        except Exception as e:
            logger.warning(f"pgvector probe failed — using in-Python retrieval: {e}")

