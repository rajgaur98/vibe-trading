"""Dead-man's-switch ping for a scheduled run. Pings HEALTHCHECK_PING_URL on a
successful run; if a run is ever missed or errors, the external monitor (e.g.
healthchecks.io) fires an alert. No-op unless the URL is configured. Never raises —
monitoring must never break trading."""
import json
import logging
import os
import urllib.request

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


def send_discord(message: str) -> None:
    """Post to the configured Discord webhook. No-op when unconfigured; never raises.
    The single Discord implementation — the scheduler delegates here."""
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url or "your_discord_webhook_url" in webhook_url:
        return
    data = json.dumps({"content": message}).encode("utf-8")
    req = urllib.request.Request(
        webhook_url, data=data,
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req):
            pass
    except Exception as e:
        logger.error(f"Failed to send Discord alert: {e}")
