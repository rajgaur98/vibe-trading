"""Security gate for the dashboard API.

Two rules, enforced as a single ASGI middleware:

1. ``POST /api/trigger-tick`` is **localhost-only**. It builds a TradingScheduler and
   runs a full evaluation (LLM spend + can place demo trades), so it must never be
   callable from the public internet. Under Docker port-mapping, host-originated traffic
   reaches the app as the bridge gateway (a private RFC1918 address) while internet
   traffic carries a real public IP — so "local" == loopback or private source.

2. Every other ``/api/*`` route requires the header ``x-api-key == DASHBOARD_API_KEY``
   **when that env var is set**. The Vercel dashboard injects the header server-side
   (so it never reaches the browser). When the env var is unset the API stays open —
   preserving local-dev behaviour and making this safe to ship before the key is rolled out.
"""
import os
import ipaddress

from fastapi import Request
from fastapi.responses import JSONResponse

TRIGGER_PATH = "/api/trigger-tick"


def is_local_client(host) -> bool:
    """True when the request comes from the host itself / a private network (not the
    public internet). Loopback and any RFC1918 private address count as local."""
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host == "localhost"
    return ip.is_loopback or ip.is_private


async def security_gate(request: Request, call_next):
    # Never block CORS preflight on the key check.
    if request.method == "OPTIONS":
        return await call_next(request)

    path = request.url.path
    if path == TRIGGER_PATH:
        client_host = request.client.host if request.client else None
        if not is_local_client(client_host):
            return JSONResponse(
                status_code=403,
                content={"detail": "trigger-tick is restricted to localhost"},
            )
    elif path.startswith("/api/"):
        api_key = os.getenv("DASHBOARD_API_KEY")
        if api_key and request.headers.get("x-api-key") != api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "missing or invalid API key"},
            )
    return await call_next(request)
