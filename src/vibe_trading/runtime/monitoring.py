"""Dead-man's-switch ping for a scheduled run. Pings HEALTHCHECK_PING_URL on a
successful run; if a run is ever missed or errors, the external monitor (e.g.
healthchecks.io) fires an alert. No-op unless the URL is configured. Never raises —
monitoring must never break trading."""
import os
import logging

import requests

logger = logging.getLogger(__name__)


def ping_healthcheck(success: bool = True) -> None:
    url = os.getenv("HEALTHCHECK_PING_URL")
    if not url:
        return
    if not success:
        url = url.rstrip("/") + "/fail"
    try:
        requests.get(url, timeout=10)
    except Exception as e:
        logger.warning(f"monitoring: healthcheck ping failed ({e})")
