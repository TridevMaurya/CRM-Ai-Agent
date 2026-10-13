"""Outbound dialing through Exotel's Connect API.

Exotel dials the user (`From`), and on answer runs the agent's flow (`app_id`),
whose Voicebot applet opens our WebSocket. Per-call context can't ride along on
the WS URL (Exotel strips query params), so the dashboard stores it as pending
context keyed by phone number, and the WS handler picks it up from the `from`
number in the start event.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request


def configured() -> bool:
    return all(os.getenv(k) for k in ("EXOTEL_API_KEY", "EXOTEL_API_TOKEN", "EXOTEL_ACCOUNT_SID"))


def dial(number: str, app_id: str, caller_id: str) -> tuple[bool, str]:
    """Returns (ok, call_sid_or_error)."""
    sid = os.environ["EXOTEL_ACCOUNT_SID"]
    sub = os.getenv("EXOTEL_SUBDOMAIN", "api.exotel.com")
    url = f"https://{sub}/v1/Accounts/{sid}/Calls/connect.json"
    data = urllib.parse.urlencode({
        "From": number,
        "CallerId": caller_id,
        "Url": f"http://my.exotel.com/{sid}/exoml/start_voice/{app_id}",
    }).encode()
    auth = base64.b64encode(
        f"{os.environ['EXOTEL_API_KEY']}:{os.environ['EXOTEL_API_TOKEN']}".encode()).decode()
    req = urllib.request.Request(url, data=data, headers={"Authorization": f"Basic {auth}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.load(resp)
            return True, body.get("Call", {}).get("Sid", "")
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.read().decode()[:300]}"
    except Exception as e:
        return False, str(e)


def _auth_header() -> dict:
    auth = base64.b64encode(
        f"{os.environ['EXOTEL_API_KEY']}:{os.environ['EXOTEL_API_TOKEN']}".encode()).decode()
    return {"Authorization": f"Basic {auth}"}


def call_details(call_sid: str) -> dict:
    """Exotel's record of a call (Status, Duration, RecordingUrl, …)."""
    sid = os.environ["EXOTEL_ACCOUNT_SID"]
    sub = os.getenv("EXOTEL_SUBDOMAIN", "api.exotel.com")
    req = urllib.request.Request(f"https://{sub}/v1/Accounts/{sid}/Calls/{call_sid}.json",
                                 headers=_auth_header())
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp).get("Call", {})


def fetch_recording(url: str) -> bytes:
    """The call recording (mp3) — Exotel keeps it, we only stream it through."""
    req = urllib.request.Request(url, headers=_auth_header())
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()
