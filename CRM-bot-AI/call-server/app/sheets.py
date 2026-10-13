"""Google Sheets access (Sheets API v4) with the same GCP service account as the TTS.

The sheet must be shared with the service account's email — Viewer to read,
Editor if we also write the call log. Blocking calls; run them in a thread.

A local .csv path works in place of a sheet id (dummy / test data): it is read as
the only tab, and rows "appended" to it go to <name>.<tab>.csv in data/.
"""

from __future__ import annotations

import csv
import os
import re
import urllib.parse
from pathlib import Path

from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
API = "https://sheets.googleapis.com/v4/spreadsheets"

ROOT = Path(__file__).resolve().parent.parent
_session: AuthorizedSession | None = None


def is_local(sheet_id: str) -> bool:
    return sheet_id.lower().endswith(".csv")


def _local(sheet_id: str) -> Path:
    path = Path(sheet_id)
    return path if path.is_absolute() else ROOT / path


def _local_out(sheet_id: str, tab: str) -> Path:
    return ROOT / "data" / f"{_local(sheet_id).stem}.{tab}.csv"


def _http() -> AuthorizedSession:
    global _session
    if _session is None:
        path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "gcp-service-account.json"
        creds = service_account.Credentials.from_service_account_file(path, scopes=SCOPES)
        _session = AuthorizedSession(creds)
    return _session


def service_account_email() -> str:
    try:
        return _http().credentials.service_account_email
    except Exception:
        return ""


def sheet_id_from(value: str) -> str:
    """Accept a bare id, a full docs.google.com URL, or a local .csv path."""
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", value or "")
    return m.group(1) if m else (value or "").strip()


def _check(resp):
    if resp.status_code >= 400:
        raise RuntimeError(f"Sheets API {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def read_values(sheet_id: str, tab: str) -> list[list[str]]:
    if is_local(sheet_id):
        with open(_local(sheet_id), newline="", encoding="utf-8-sig") as f:
            return list(csv.reader(f))
    rng = urllib.parse.quote(f"'{tab}'", safe="")
    data = _check(_http().get(f"{API}/{sheet_id}/values/{rng}",
                              params={"valueRenderOption": "FORMATTED_VALUE"}, timeout=60))
    return data.get("values", [])


def tab_names(sheet_id: str) -> list[str]:
    if is_local(sheet_id):
        return [_local(sheet_id).stem]
    data = _check(_http().get(f"{API}/{sheet_id}", params={"fields": "sheets.properties.title"},
                              timeout=30))
    return [s["properties"]["title"] for s in data.get("sheets", [])]


def ensure_tab(sheet_id: str, tab: str, header: list[str]) -> None:
    if is_local(sheet_id):
        if not _local_out(sheet_id, tab).exists():
            append_rows(sheet_id, tab, [header])
        return
    if tab in tab_names(sheet_id):
        return
    _check(_http().post(f"{API}/{sheet_id}:batchUpdate", timeout=30,
                        json={"requests": [{"addSheet": {"properties": {"title": tab}}}]}))
    append_rows(sheet_id, tab, [header])


def append_rows(sheet_id: str, tab: str, rows: list[list]) -> None:
    if is_local(sheet_id):
        out = _local_out(sheet_id, tab)
        out.parent.mkdir(exist_ok=True)
        with open(out, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        return
    rng = urllib.parse.quote(f"'{tab}'!A1", safe="")
    _check(_http().post(f"{API}/{sheet_id}/values/{rng}:append", timeout=30,
                        params={"valueInputOption": "RAW", "insertDataOption": "INSERT_ROWS"},
                        json={"values": [[("" if v is None else v) for v in r] for r in rows]}))


def norm_header(h: str) -> str:
    """'Video-Title' → 'video_title', 'Content ID' → 'content_id'."""
    return re.sub(r"[^a-z0-9]+", "_", (h or "").strip().lower()).strip("_")


def read_records(sheet_id: str, tab: str, must_have: str | tuple[str, ...]) -> list[dict]:
    """Rows as dicts keyed by normalised header. The header is the first row (of the
    first 10) that contains `must_have` (or any of several names) — so title / note
    rows above it are fine."""
    names = (must_have,) if isinstance(must_have, str) else must_have
    values = read_values(sheet_id, tab)
    for i, row in enumerate(values[:10]):
        header = [norm_header(h) for h in row]
        if any(n in header for n in names):
            out = []
            for r in values[i + 1:]:
                rec = {h: (r[j].strip() if j < len(r) else "") for j, h in enumerate(header) if h}
                if any(rec.values()):
                    out.append(rec)
            return out
    raise RuntimeError(f"tab {tab!r}: no header row with a {' / '.join(names)!r} column in the first 10 rows")
