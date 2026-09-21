"""Unit tests for PostgresDatabase connection-pool resilience.

Reproduces the recurring prod incident where Supabase server-side-closes idle
pooled connections and the API's ThreadedConnectionPool hands the dead
connections straight back out (poisoning the pool -> dashboard 500s), plus the
chronic latency from re-running _create_tables() DDL on every request.

These tests use a fake pool and never touch a real database, so they run the
same whether or not POSTGRES_URL is set.
"""
import psycopg2
import pytest

from vibe_trading.data.db import PostgresDatabase, PostgresConnectionWrapper


class FakeCursor:
    def __init__(self, conn):
        self._conn = conn
        self._result = None

    def execute(self, sql, params=None):
        if not self._conn.alive:
            raise psycopg2.OperationalError(
                "server closed the connection unexpectedly")
        self._conn.executed.append(sql)
        self._result = (1,)
        return self

    def fetchone(self):
        return self._result

    def close(self):
        pass


class FakeConn:
    """Minimal stand-in for a psycopg2 connection as used by connect()/close()."""
    def __init__(self, alive=True):
        self.alive = alive
        self.closed = 0  # psycopg2 sets this nonzero once a conn is broken/closed
        self.executed = []
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        if not self.alive:
            raise psycopg2.OperationalError("connection already closed")
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = 1


class FakePool:
    def __init__(self, conns):
        self._conns = list(conns)
        self.getconn_count = 0
        self.putconn_calls = []  # (conn, close_flag)

    def getconn(self):
        self.getconn_count += 1
        return self._conns.pop(0)

    def putconn(self, conn, close=False):
        self.putconn_calls.append((conn, close))


@pytest.fixture
def isolate_pg_class():
    """Save/restore the mutable class-level state PostgresDatabase carries."""
    saved_pool = PostgresDatabase._pool
    saved_pgv = PostgresDatabase.pgvector_enabled
    saved_schema = getattr(PostgresDatabase, "_schema_ready", False)
    PostgresDatabase._pool = None
    PostgresDatabase.pgvector_enabled = False
    PostgresDatabase._schema_ready = False
    try:
        yield
    finally:
        PostgresDatabase._pool = saved_pool
        PostgresDatabase.pgvector_enabled = saved_pgv
        PostgresDatabase._schema_ready = saved_schema


def _bare_db():
    """A PostgresDatabase instance without __init__ (no real pool / DDL side effects)."""
    db = PostgresDatabase.__new__(PostgresDatabase)
    db.conn = None
    db.db_url = "postgres://fake/db"
    return db


def test_connect_discards_dead_pooled_connection_and_acquires_live_one(isolate_pg_class):
    dead = FakeConn(alive=False)
    live = FakeConn(alive=True)
    PostgresDatabase._pool = FakePool([dead, live])
    db = _bare_db()

    db.connect()

    # The dead connection must be discarded (close=True), not returned alive.
    assert (dead, True) in PostgresDatabase._pool.putconn_calls
    # The caller must receive the live connection.
    assert isinstance(db.conn, PostgresConnectionWrapper)
    assert db.conn.connection is live


def test_connect_raises_after_exhausting_retries_on_all_dead_conns(isolate_pg_class):
    deads = [FakeConn(alive=False) for _ in range(5)]
    PostgresDatabase._pool = FakePool(deads)
    db = _bare_db()

    with pytest.raises(psycopg2.Error):
        db.connect()

    # Every connection it pulled was discarded, never leaked back into the pool alive.
    assert PostgresDatabase._pool.putconn_calls  # it did try
    assert all(close is True for _, close in PostgresDatabase._pool.putconn_calls)


def test_close_discards_a_broken_connection(isolate_pg_class):
    live = FakeConn(alive=True)
    PostgresDatabase._pool = FakePool([live])
    db = _bare_db()
    db.connect()

    # Simulate the connection dying mid-request (Supabase drops it server-side).
    live.alive = False
    live.closed = 2

    db.close()

    assert (live, True) in PostgresDatabase._pool.putconn_calls


def test_close_returns_a_healthy_connection_without_discarding(isolate_pg_class):
    live = FakeConn(alive=True)
    PostgresDatabase._pool = FakePool([live])
    db = _bare_db()
    db.connect()

    db.close()

    # Healthy conn returned to the pool for reuse (close=False).
    assert (live, False) in PostgresDatabase._pool.putconn_calls


def test_create_tables_is_noop_once_schema_ready(isolate_pg_class):
    PostgresDatabase._schema_ready = True
    # Empty pool: any attempt to connect/getconn would IndexError.
    PostgresDatabase._pool = FakePool([])
    db = _bare_db()

    db._create_tables()  # must short-circuit without touching the pool

    assert PostgresDatabase._pool.getconn_count == 0


def test_pool_initialized_with_tcp_keepalives(isolate_pg_class, monkeypatch):
    captured = {}

    def fake_pool_ctor(minconn, maxconn, dsn=None, **kwargs):
        captured["args"] = (minconn, maxconn, dsn)
        captured["kwargs"] = kwargs
        return FakePool([])

    monkeypatch.setattr("vibe_trading.data.db.ThreadedConnectionPool", fake_pool_ctor)
    PostgresDatabase._pool = None
    db = _bare_db()

    db._initialize_pool()

    kwargs = captured["kwargs"]
    assert kwargs.get("keepalives") == 1
    assert "keepalives_idle" in kwargs
    assert "keepalives_interval" in kwargs
    assert "keepalives_count" in kwargs
