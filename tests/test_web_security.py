"""Tests for the dashboard API security gate (web/security.py):
- /api/trigger-tick is restricted to local (loopback/private) clients.
- all other /api/* routes require x-api-key == DASHBOARD_API_KEY when that env is set.
The gate is mounted on a throwaway FastAPI app so no Postgres/import side effects occur."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vibe_trading.web import security


def _app():
    app = FastAPI()
    app.middleware("http")(security.security_gate)

    @app.get("/api/metrics")
    def metrics():
        return {"ok": True}

    @app.post("/api/trigger-tick")
    def trigger():
        return {"ok": True}

    return app


@pytest.mark.parametrize("host,expected", [
    ("127.0.0.1", True),
    ("::1", True),
    ("172.18.0.1", True),     # docker bridge gateway = host-originated (localhost)
    ("10.0.0.5", True),       # private VCN
    ("192.168.1.9", True),    # private
    ("103.189.221.56", False),  # public (a laptop)
    ("13.127.90.248", False),   # public
    ("localhost", True),
    ("", False),
    (None, False),
])
def test_is_local_client(host, expected):
    assert security.is_local_client(host) is expected


def test_trigger_tick_blocks_remote(monkeypatch):
    monkeypatch.setattr(security, "is_local_client", lambda h: False)
    r = TestClient(_app()).post("/api/trigger-tick")
    assert r.status_code == 403


def test_trigger_tick_allows_local(monkeypatch):
    monkeypatch.setattr(security, "is_local_client", lambda h: True)
    r = TestClient(_app()).post("/api/trigger-tick")
    assert r.status_code == 200


def test_reads_open_when_no_key(monkeypatch):
    monkeypatch.delenv("DASHBOARD_API_KEY", raising=False)
    assert TestClient(_app()).get("/api/metrics").status_code == 200


def test_reads_require_key_when_set(monkeypatch):
    monkeypatch.setenv("DASHBOARD_API_KEY", "secret123")
    c = TestClient(_app())
    assert c.get("/api/metrics").status_code == 401                              # no header
    assert c.get("/api/metrics", headers={"x-api-key": "wrong"}).status_code == 401
    assert c.get("/api/metrics", headers={"x-api-key": "secret123"}).status_code == 200


def test_trigger_tick_does_not_need_key_when_local(monkeypatch):
    """A local trigger works even with the key configured (local trust is sufficient)."""
    monkeypatch.setenv("DASHBOARD_API_KEY", "secret123")
    monkeypatch.setattr(security, "is_local_client", lambda h: True)
    assert TestClient(_app()).post("/api/trigger-tick").status_code == 200


def test_options_preflight_not_blocked_by_key(monkeypatch):
    monkeypatch.setenv("DASHBOARD_API_KEY", "secret123")
    r = TestClient(_app()).options("/api/metrics")
    assert r.status_code != 401   # CORS preflight must not be rejected by the key gate
