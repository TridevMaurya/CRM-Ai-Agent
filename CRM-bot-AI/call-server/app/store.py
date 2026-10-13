"""SQLite store — calls, transcript turns, post-call analysis, pending dial context,
and the campaign queue (contacts synced from the user sheet + the content catalogue).

One file (data/crm.db). Kept deliberately simple: a handful of functions, no ORM.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "crm.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id            TEXT PRIMARY KEY,          -- our id (Exotel call_sid when present)
    agent_id      TEXT NOT NULL,
    transport     TEXT NOT NULL,             -- exotel | browser
    from_number   TEXT DEFAULT '',
    to_number     TEXT DEFAULT '',
    language      TEXT DEFAULT '',
    params        TEXT DEFAULT '{}',         -- per-call context (json)
    status        TEXT DEFAULT 'live',       -- live | ended | failed
    started_at    REAL NOT NULL,
    ended_at      REAL,
    duration_secs REAL,
    cost_inr      REAL,
    cost_detail   TEXT DEFAULT '{}',
    latency_p50   REAL,
    analysis      TEXT,                      -- json from post-call extraction
    bucket        TEXT,                      -- positive | neutral | negative (see bucket_of)
    error         TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS turns (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id  TEXT NOT NULL,
    role     TEXT NOT NULL,                  -- user | bot
    text     TEXT NOT NULL,
    at       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS turns_call ON turns(call_id, id);
CREATE TABLE IF NOT EXISTS agent_defaults (
    agent_id   TEXT PRIMARY KEY,
    params     TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS pending_context (
    number     TEXT PRIMARY KEY,             -- normalised phone number
    agent_id   TEXT NOT NULL,
    params     TEXT NOT NULL,
    created_at REAL NOT NULL
);
-- One row per (phone, cohort), synced from the user sheet. The dialer walks the
-- `queued` rows; the call handler and the post-call step move them along.
CREATE TABLE IF NOT EXISTS contacts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    phone           TEXT NOT NULL,           -- normalised (last 10 digits)
    cohort          TEXT NOT NULL,
    agent_id        TEXT NOT NULL,
    params          TEXT NOT NULL,           -- everything the prompt needs (json)
    status          TEXT NOT NULL DEFAULT 'queued',
                    -- queued | dialing | in_call | done | no_answer | failed | dnc | skipped
    attempts        INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL NOT NULL DEFAULT 0,
    last_dialed_at  REAL,
    call_id         TEXT DEFAULT '',
    note            TEXT DEFAULT '',
    synced_at       REAL NOT NULL,
    created_at      REAL NOT NULL,
    UNIQUE (phone, cohort)
);
CREATE INDEX IF NOT EXISTS contacts_due ON contacts(status, next_attempt_at);
CREATE INDEX IF NOT EXISTS contacts_phone ON contacts(phone);
-- Content catalogue from the content sheet: content_id → title / category / summary.
CREATE TABLE IF NOT EXISTS content (
    content_id TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    category   TEXT DEFAULT '',
    language   TEXT DEFAULT '',
    script     TEXT DEFAULT '',
    summary    TEXT DEFAULT '',              -- ~10-second spoken summary (from script)
    synced_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
-- Admin-panel edits layered over the agent YAML (nested like the YAML), applied on
-- the next call without a deploy.
CREATE TABLE IF NOT EXISTS agent_overrides (
    agent_id   TEXT PRIMARY KEY,
    overrides  TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""

PENDING_TTL_SECS = 2 * 3600


def init() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with _conn() as c:
        c.executescript(SCHEMA)
        cols = {r["name"] for r in c.execute("PRAGMA table_info(calls)")}
        if "bucket" not in cols:
            c.execute("ALTER TABLE calls ADD COLUMN bucket TEXT")
    backfill_buckets()
    backfill_themes()
    close_orphans()


def close_orphans() -> tuple[int, int]:
    """A restart (deploy, crash) kills every call this process was running: mark them
    failed instead of leaving them "live" forever, and put their contacts back in the
    queue so they get their retry."""
    now = time.time()
    with _conn() as c:
        calls = c.execute("SELECT id, started_at FROM calls WHERE status='live'").fetchall()
        for r in calls:
            c.execute("UPDATE calls SET status='failed', error='server restarted during call', "
                      "ended_at=?, duration_secs=? WHERE id=?", (now, max(0, now - r["started_at"]), r["id"]))
        n = c.execute("UPDATE contacts SET status='queued', next_attempt_at=? "
                      "WHERE status IN ('in_call', 'dialing')", (now,)).rowcount
    return len(calls), n


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def normalize_number(num: str) -> str:
    digits = "".join(ch for ch in (num or "") if ch.isdigit())
    return digits[-10:] if len(digits) >= 10 else digits


# ----- calls -----

def create_call(call_id: str, agent_id: str, transport: str, *, from_number="", to_number="",
                language="", params: dict | None = None) -> None:
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO calls (id, agent_id, transport, from_number, to_number, "
            "language, params, status, started_at) VALUES (?,?,?,?,?,?,?,'live',?)",
            (call_id, agent_id, transport, from_number, to_number, language,
             json.dumps(params or {}), time.time()),
        )


def add_turn(call_id: str, role: str, text: str) -> None:
    with _conn() as c:
        c.execute("INSERT INTO turns (call_id, role, text, at) VALUES (?,?,?,?)",
                  (call_id, role, text, time.time()))


def end_call(call_id: str, *, status="ended", cost: dict | None = None,
             latency_p50: float | None = None, error: str = "") -> None:
    cost = cost or {}
    with _conn() as c:
        c.execute(
            "UPDATE calls SET status=?, ended_at=?, duration_secs=? - started_at, cost_inr=?, "
            "cost_detail=?, latency_p50=?, error=? WHERE id=?",
            (status, time.time(), time.time(), cost.get("total_inr"), json.dumps(cost),
             latency_p50, error, call_id),
        )


# "Positive" = the call produced a real content insight (what to add, what to fix,
# what they liked/disliked and why). The analysis model decides (has_content_insight);
# calls analysed before that field existed fall back to a strict text heuristic.
_INSIGHT_FIELDS = ("praise", "complaints", "story_content", "presentation_feedback", "improve_video",
                   "wants_more_of", "missing_content", "video_dislike", "one_improvement",
                   "wanted_video", "proof_needed", "other_apps_gap",
                   "likes_most_about_gyantv", "why_this_category", "would_have_continued_if")
_EMPTY = {"", "unclear", "none", "n/a", "not discussed", "not mentioned", "no", "false", "empty", "nothing"}
_NOT_REALLY = ("did not", "didn't", "no specific", "not provide", "unable to", "no feedback", "not discussed",
               "not mention", "no clear", "could not", "not give", "declined", "refund", "no actionable")


def _substantive(v) -> bool:
    if isinstance(v, list):
        return any(_substantive(x) for x in v)
    t = str(v or "").strip()
    low = t.lower()
    return (low not in _EMPTY and len(t) >= 6 and not any(k in low for k in _NOT_REALLY))


def bucket_of(analysis: dict | None) -> str:
    """positive = real content insight; negative = no insight and a negative mood
    (refund, angry, do-not-call); neutral = nothing usable (silent, busy, vague)."""
    if not analysis:
        return "none"
    flag = analysis.get("has_content_insight")
    if flag is None:
        insight = any(_substantive(analysis.get(k)) for k in _INSIGHT_FIELDS)
    else:
        insight = str(flag).strip().lower() == "true"
    if insight:
        return "positive"
    if str(analysis.get("sentiment", "")).lower() == "negative" or analysis.get("do_not_call") is True:
        return "negative"
    return "neutral"


def set_analysis(call_id: str, analysis: dict) -> None:
    with _conn() as c:
        c.execute("UPDATE calls SET analysis=?, bucket=? WHERE id=?",
                  (json.dumps(analysis), bucket_of(analysis), call_id))
    retag_call(call_id, analysis)


def retag_call(call_id: str, analysis: dict | None = None, turns: list[dict] | None = None) -> str:
    """Theme the call by what was actually discussed: the analysis' theme_discussed
    field, else a UPSC/PSC mention in the transcript. Added to params.themes and, for a
    priority theme, made the primary theme."""
    from .themes import norm_theme, theme_mentioned, PRIORITY_THEMES
    with _conn() as c:
        r = c.execute("SELECT params, analysis FROM calls WHERE id=?", (call_id,)).fetchone()
        if not r:
            return ""
        try:
            p = json.loads(r["params"] or "{}")
        except (TypeError, ValueError):
            p = {}
        a = analysis if analysis is not None else (json.loads(r["analysis"]) if r["analysis"] else {})
        discussed = norm_theme(str(a.get("theme_discussed") or ""))
        if not discussed:
            text = " ".join([str(a.get("summary") or ""), str(a.get("reason_detail") or "")]
                            + [t.get("text", "") for t in (turns or []) if t.get("role") == "user"])
            discussed = theme_mentioned(text)
        if not discussed or discussed.lower() in ("unknown", "none", "unclear"):
            return ""
        themes = list(p.get("themes") or [])
        if p.get("theme") and p["theme"] not in themes:
            themes.insert(0, p["theme"])
        if discussed not in themes:
            themes.append(discussed)
        if discussed in PRIORITY_THEMES or not p.get("theme"):
            p["theme"] = discussed
        p["themes"] = themes
        c.execute("UPDATE calls SET params=? WHERE id=?", (json.dumps(p, ensure_ascii=False), call_id))
    return discussed


def backfill_themes() -> int:
    """Give every call and contact a normalised params.theme (older rows only carry
    the sheet's raw top_category / video_category); then re-tag analysed calls by what
    was discussed (once — marked with params.retagged)."""
    from .themes import norm_theme
    n = 0
    with _conn() as c:
        todo = c.execute("SELECT id FROM calls WHERE analysis IS NOT NULL AND "
                         "json_extract(params, '$.retagged') IS NULL").fetchall()
    for r in todo:
        call = get_call(r["id"])
        retag_call(r["id"], None, call["turns"] if call else [])
        with _conn() as c:
            c.execute("UPDATE calls SET params=json_set(params, '$.retagged', 1) WHERE id=?", (r["id"],))
    with _conn() as c:
        for table in ("calls", "contacts"):
            rows = c.execute(f"SELECT id, params FROM {table} WHERE {THEME_SQL} IS NULL "
                             "AND params IS NOT NULL").fetchall()
            for r in rows:
                try:
                    p = json.loads(r["params"] or "{}")
                except (TypeError, ValueError):
                    continue
                theme = norm_theme(p.get("top_category") or p.get("video_category"))
                if not theme:
                    continue
                p["theme"] = theme
                c.execute(f"UPDATE {table} SET params=? WHERE id=?", (json.dumps(p, ensure_ascii=False), r["id"]))
                n += 1
    return n


def backfill_buckets() -> int:
    """Re-classify every analysed call with the current bucket rule."""
    n = 0
    with _conn() as c:
        for r in c.execute("SELECT id, analysis FROM calls WHERE analysis IS NOT NULL").fetchall():
            try:
                a = json.loads(r["analysis"])
            except (TypeError, ValueError):
                continue
            c.execute("UPDATE calls SET bucket=? WHERE id=?", (bucket_of(a), r["id"])); n += 1
    return n


def _row_to_call(r: sqlite3.Row) -> dict:
    d = dict(r)
    for key in ("params", "cost_detail", "analysis"):
        try:
            d[key] = json.loads(d[key]) if d.get(key) else ({} if key != "analysis" else None)
        except (TypeError, ValueError):
            d[key] = None
    return d


THEME_SQL = "json_extract(params, '$.theme')"


def _calls_where(agent_id: str | None, sentiment: str | None, transport: str | None,
                 since: float | None = None, cohort: str | None = None, theme: str | None = None):
    where, args = [], []
    if theme:
        # primary theme, or any theme in params.themes (a UPSC + Finance viewer shows under both)
        where.append(f"({THEME_SQL}=? OR EXISTS (SELECT 1 FROM json_each(calls.params, '$.themes') "
                     f"WHERE json_each.value=?))")
        args += [theme, theme]
    if agent_id:
        where.append("agent_id=?"); args.append(agent_id)
    if transport:
        where.append("transport=?"); args.append(transport)
    if since:
        where.append("started_at>=?"); args.append(since)
    if cohort:
        where.append("json_extract(params, '$.cohort')=?"); args.append(cohort)
    if sentiment == "none":
        where.append("bucket IS NULL")
    elif sentiment:
        where.append("bucket=?"); args.append(sentiment)
    return (" WHERE " + " AND ".join(where)) if where else "", args


def list_calls(limit: int = 100, agent_id: str | None = None, sentiment: str | None = None,
               transport: str | None = None, offset: int = 0, since: float | None = None,
               cohort: str | None = None, theme: str | None = None) -> list[dict]:
    """sentiment: positive | neutral | negative | none (no analysis yet)."""
    where, args = _calls_where(agent_id, sentiment, transport, since, cohort, theme)
    with _conn() as c:
        rows = c.execute(f"SELECT * FROM calls{where} ORDER BY started_at DESC LIMIT ? OFFSET ?",
                         (*args, limit, offset)).fetchall()
    return [_row_to_call(r) for r in rows]


def count_calls(agent_id: str | None = None, sentiment: str | None = None,
                transport: str | None = None, since: float | None = None, cohort: str | None = None,
                theme: str | None = None) -> int:
    where, args = _calls_where(agent_id, sentiment, transport, since, cohort, theme)
    with _conn() as c:
        return c.execute(f"SELECT COUNT(*) FROM calls{where}", args).fetchone()[0]


def sentiment_counts(transport: str | None = None) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT agent_id, COALESCE(bucket, 'none') s, COUNT(*) n "
            "FROM calls" + (" WHERE transport=?" if transport else "") + " GROUP BY agent_id, s",
            ((transport,) if transport else ())).fetchall()
    return [dict(r) for r in rows]


def get_call(call_id: str) -> dict | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM calls WHERE id=?", (call_id,)).fetchone()
        if not r:
            return None
        turns = c.execute("SELECT role, text, at FROM turns WHERE call_id=? ORDER BY id",
                          (call_id,)).fetchall()
    call = _row_to_call(r)
    call["turns"] = [dict(t) for t in turns]
    return call


def insights(agent_id: str | None, since: float | None, cohort: str | None,
             transport: str | None = "exotel", theme: str | None = None) -> dict:
    """Everything the Calls page shows above the table for one selection. `themes`
    ignores the theme filter so the filter list always shows every theme."""
    from collections import Counter
    where, args = _calls_where(agent_id, None, transport, since, cohort)
    with _conn() as c:
        rows = c.execute(f"SELECT bucket, analysis, duration_secs, cost_inr, status, params, "
                         f"{THEME_SQL} AS theme FROM calls{where}", args).fetchall()
    theme_n, theme_pos = Counter(), Counter()

    def _themes(r) -> list[str]:
        try:
            p = json.loads(r["params"] or "{}")
        except (TypeError, ValueError):
            p = {}
        ts = list(p.get("themes") or [])
        if p.get("theme") and p["theme"] not in ts:
            ts.insert(0, p["theme"])
        return ts or ["Unknown"]
    for r in rows:
        for t in _themes(r):
            theme_n[t] += 1
            if r["bucket"] == "positive":
                theme_pos[t] += 1
    themes = [(t, n, theme_pos[t]) for t, n in theme_n.most_common()]
    if theme:
        rows = [r for r in rows if theme in _themes(r)]
    buckets = Counter((r["bucket"] or "none") for r in rows)
    reasons, wants, missing, apps = Counter(), Counter(), Counter(), Counter()
    prices = []
    for r in rows:
        try:
            a = json.loads(r["analysis"]) if r["analysis"] else {}
        except (TypeError, ValueError):
            a = {}
        pr = str(a.get("primary_reason") or "").strip().lower()
        if pr and pr != "unclear":
            reasons[pr] += 1
        for key, ctr in (("wants_more_of", wants), ("missing_content", missing), ("other_apps", apps)):
            v = a.get(key)
            for x in (v if isinstance(v, list) else [v]):
                x = str(x or "").strip()
                if x and x.lower() not in ("unclear", "none", "empty", "[]"):
                    ctr[x[:60]] += 1
        if a.get("willing_price"):
            prices.append(str(a["willing_price"])[:30])
    n = len(rows)
    durs = [r["duration_secs"] or 0 for r in rows]
    return {
        "n": n, "live": sum(1 for r in rows if r["status"] == "live"),
        "buckets": {k: buckets.get(k, 0) for k in ("positive", "neutral", "negative", "none")},
        "avg_secs": (sum(durs) / n) if n else 0, "talked_1min": sum(1 for d in durs if d >= 60),
        "cost": sum(r["cost_inr"] or 0 for r in rows),
        "reasons": reasons.most_common(6), "wants": wants.most_common(6), "missing": missing.most_common(6),
        "apps": apps.most_common(6), "prices": Counter(prices).most_common(6),
        "themes": themes,
    }


def kpis(since: float) -> dict:
    with _conn() as c:
        r = c.execute(
            "SELECT COUNT(*) n, SUM(status='live') live, "
            "SUM(json_extract(analysis, '$.summary') IS NOT NULL) analysed, "
            "SUM(bucket='positive') useful, SUM(duration_secs>=60) long, AVG(duration_secs) avg_secs, "
            "SUM(cost_inr) cost FROM calls WHERE transport='exotel' AND started_at>=?", (since,)).fetchone()
    return {k: (r[k] or 0) for k in r.keys()}


def agent_stats() -> dict[str, dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT agent_id, COUNT(*) n, SUM(status='live') live, "
            "AVG(duration_secs) avg_secs, SUM(cost_inr) cost FROM calls GROUP BY agent_id"
        ).fetchall()
    return {r["agent_id"]: dict(r) for r in rows}


# ----- per-agent default context (used when a call has no number-specific context,
#       e.g. someone calls the ExoPhone themselves) -----

def set_agent_default(agent_id: str, params: dict) -> None:
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO agent_defaults (agent_id, params, updated_at) VALUES (?,?,?)",
                  (agent_id, json.dumps(params), time.time()))


def get_agent_default(agent_id: str) -> dict:
    with _conn() as c:
        r = c.execute("SELECT params FROM agent_defaults WHERE agent_id=?", (agent_id,)).fetchone()
    return json.loads(r["params"]) if r else {}


# ----- pending dial context (number → agent + params, consumed on connect) -----

def put_pending(number: str, agent_id: str, params: dict) -> None:
    with _conn() as c:
        c.execute("DELETE FROM pending_context WHERE created_at < ?", (time.time() - PENDING_TTL_SECS,))
        c.execute("INSERT OR REPLACE INTO pending_context (number, agent_id, params, created_at) "
                  "VALUES (?,?,?,?)", (normalize_number(number), agent_id, json.dumps(params), time.time()))


def pop_pending(number: str) -> tuple[str, dict] | None:
    key = normalize_number(number)
    if not key:
        return None
    with _conn() as c:
        r = c.execute("SELECT agent_id, params, created_at FROM pending_context WHERE number=?",
                      (key,)).fetchone()
        if not r:
            return None
        c.execute("DELETE FROM pending_context WHERE number=?", (key,))
    if time.time() - r["created_at"] > PENDING_TTL_SECS:
        return None
    return r["agent_id"], json.loads(r["params"])


# ----- content catalogue -----

def upsert_content(rows: list[dict]) -> None:
    """Replace the catalogue with `rows` (content_id, title, category, language, script,
    summary). A generated summary is kept while the script hasn't changed, so it is
    only made once. Items no longer in the sheet are removed."""
    now = time.time()
    with _conn() as c:
        ids = [r["content_id"] for r in rows]
        c.execute("CREATE TEMP TABLE keep (id TEXT PRIMARY KEY)")
        c.executemany("INSERT OR IGNORE INTO keep VALUES (?)", [(i,) for i in ids])
        c.execute("DELETE FROM content WHERE content_id NOT IN (SELECT id FROM keep)")
        for r in rows:
            old = c.execute("SELECT script, summary FROM content WHERE content_id=?",
                            (r["content_id"],)).fetchone()
            summary = r.get("summary") or ""
            if not summary and old and old["script"] == r.get("script", ""):
                summary = old["summary"]
            c.execute(
                "INSERT OR REPLACE INTO content (content_id, title, category, language, script, "
                "summary, synced_at) VALUES (?,?,?,?,?,?,?)",
                (r["content_id"], r["title"], r.get("category", ""), r.get("language", ""),
                 r.get("script", ""), summary, now),
            )


def content_needing_summary(limit: int = 50) -> list[dict]:
    with _conn() as c:
        # Hindi first: the pilot dials Hindi contacts only (DIAL_LANGUAGES=hin).
        rows = c.execute("SELECT content_id, title, script FROM content WHERE script != '' "
                         "AND summary = '' ORDER BY CASE WHEN language = 'Hindi' THEN 0 ELSE 1 END, "
                         "content_id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def set_content_summary(content_id: str, summary: str) -> None:
    with _conn() as c:
        c.execute("UPDATE content SET summary=? WHERE content_id=?", (summary, content_id))


def get_content(content_id: str) -> dict | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM content WHERE content_id=?", (str(content_id).strip(),)).fetchone()
    return dict(r) if r else None


def content_count() -> int:
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM content").fetchone()[0]


# ----- campaign queue -----

def upsert_contacts(rows: list[tuple[str, str, str, dict]], recall_after_secs: float) -> dict:
    """Bulk sync of (phone, cohort, agent_id, params) in one transaction: insert new
    contacts, refresh the params of ones still waiting, re-queue ones finished long
    enough ago (`recall_after_secs`); leave the rest. Returns counts."""
    now = time.time()
    counts = {"new": 0, "updated": 0, "requeued": 0, "skipped": 0}
    with _conn() as c:
        existing = {(r["phone"], r["cohort"]): r for r in
                    c.execute("SELECT id, phone, cohort, status, last_dialed_at FROM contacts")}
        for phone, cohort, agent_id, params in rows:
            phone = normalize_number(phone)
            r = existing.get((phone, cohort))
            blob = json.dumps(params, ensure_ascii=False)
            if not r:
                c.execute("INSERT INTO contacts (phone, cohort, agent_id, params, synced_at, created_at) "
                          "VALUES (?,?,?,?,?,?)", (phone, cohort, agent_id, blob, now, now))
                counts["new"] += 1
            elif r["status"] == "queued":
                c.execute("UPDATE contacts SET params=?, agent_id=?, synced_at=? WHERE id=?",
                          (blob, agent_id, now, r["id"]))
                counts["updated"] += 1
            elif (r["status"] in ("done", "no_answer", "failed")
                  and (r["last_dialed_at"] or 0) < now - recall_after_secs):
                c.execute("UPDATE contacts SET params=?, agent_id=?, status='queued', attempts=0, "
                          "next_attempt_at=0, call_id='', note='', synced_at=? WHERE id=?",
                          (blob, agent_id, now, r["id"]))
                counts["requeued"] += 1
            else:
                counts["skipped"] += 1
    return counts


def all_content() -> dict[str, dict]:
    with _conn() as c:
        return {r["content_id"]: dict(r) for r in c.execute("SELECT * FROM content")}


def phone_is_dnc(phone: str) -> bool:
    with _conn() as c:
        return c.execute("SELECT 1 FROM contacts WHERE phone=? AND status='dnc' LIMIT 1",
                         (normalize_number(phone),)).fetchone() is not None


def recently_called(phone: str, within_secs: float, exclude_id: int) -> bool:
    """Did we talk to this number (any cohort) recently?"""
    with _conn() as c:
        return c.execute("SELECT 1 FROM contacts WHERE phone=? AND id != ? AND status='done' "
                         "AND last_dialed_at > ? LIMIT 1",
                         (normalize_number(phone), exclude_id, time.time() - within_secs)).fetchone() is not None


def next_due_contact(languages: tuple[str, ...] = (), cohorts: tuple[str, ...] = (),
                     themes_first: tuple[str, ...] = (), video_category: str | None = None) -> dict | None:
    """`languages`: only these (params.language). `cohorts`: only these, and in this
    order of priority; empty = all, SUB > TRIAL > DAU > LIGHT. `themes_first`: contacts
    whose theme is one of these go before everything else, whatever the cohort."""
    lang_sql = " AND (" + " OR ".join("params LIKE ?" for _ in languages) + ")" if languages else ""
    args = [time.time()] + [f'%"language": "{l}"%' for l in languages]
    order = list(cohorts) or ["SUB_CANCELLED", "TRIAL_CANCELLED", "DAU_10PLUS", "LIGHT_3_4"]
    cohort_sql = (" AND cohort IN (" + ",".join("?" * len(cohorts)) + ")") if cohorts else ""
    args += list(cohorts)
    if video_category:          # only people whose LAST video was in this category
        cohort_sql += " AND json_extract(params, '$.video_category')=?"
        args.append(video_category)
    case = "CASE cohort " + " ".join(f"WHEN '{c}' THEN {i}" for i, c in enumerate(order)) + " ELSE 99 END"
    theme_rank = ""     # a bare literal in ORDER BY would be read as a column index
    if themes_first:
        theme_rank = f"CASE WHEN {THEME_SQL} IN ({','.join('?' * len(themes_first))}) THEN 0 ELSE 1 END, "
    with _conn() as c:
        # Fresh contacts before retries, then priority themes, then cohort priority.
        r = c.execute("SELECT * FROM contacts WHERE status='queued' AND next_attempt_at <= ?"
                      + lang_sql + cohort_sql
                      + f" ORDER BY attempts, {theme_rank}{case}, created_at, id LIMIT 1",
                      args + list(themes_first)).fetchone()
    return _row_to_contact(r) if r else None


def _row_to_contact(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["params"] = json.loads(d["params"] or "{}")
    return d


def get_contact(contact_id: int) -> dict | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM contacts WHERE id=?", (contact_id,)).fetchone()
    return _row_to_contact(r) if r else None


def latest_contact_for_phone(phone: str) -> dict | None:
    """The contact we most recently dialed (or queued) on this number — used to give
    an incoming call (a user returning our missed call) the right agent + context."""
    with _conn() as c:
        r = c.execute("SELECT * FROM contacts WHERE phone=? ORDER BY COALESCE(last_dialed_at, 0) DESC, "
                      "id DESC LIMIT 1", (normalize_number(phone),)).fetchone()
    return _row_to_contact(r) if r else None


def update_contact(contact_id: int, **fields) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    with _conn() as c:
        c.execute(f"UPDATE contacts SET {cols} WHERE id=?", (*fields.values(), contact_id))


def stale_dialing(older_than_secs: float) -> list[dict]:
    """Contacts we dialed that never connected to our WebSocket (no answer / busy)."""
    with _conn() as c:
        rows = c.execute("SELECT * FROM contacts WHERE status='dialing' AND last_dialed_at < ?",
                         (time.time() - older_than_secs,)).fetchall()
    return [_row_to_contact(r) for r in rows]


def count_in_call() -> int:
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM contacts WHERE status='in_call'").fetchone()[0]


def count_in_flight() -> int:
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM contacts WHERE status IN ('dialing','in_call')").fetchone()[0]


def queue_summary() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT cohort, status, COUNT(*) n FROM contacts GROUP BY cohort, status "
                         "ORDER BY cohort, status").fetchall()
    return [dict(r) for r in rows]


def list_contacts(limit: int = 200, status: str | None = None) -> list[dict]:
    with _conn() as c:
        if status:
            rows = c.execute("SELECT * FROM contacts WHERE status=? ORDER BY COALESCE(last_dialed_at, "
                             "created_at) DESC LIMIT ?", (status, limit)).fetchall()
        else:
            rows = c.execute("SELECT * FROM contacts ORDER BY COALESCE(last_dialed_at, created_at) "
                             "DESC LIMIT ?", (limit,)).fetchall()
    return [_row_to_contact(r) for r in rows]


# ----- key/value settings (dialer on/off, last sync) -----

def get_setting(key: str, default: str = "") -> str:
    with _conn() as c:
        r = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return r["value"] if r else default


def set_setting(key: str, value: str) -> None:
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?,?)", (key, value))


# ----- admin-panel overrides over the agent YAML -----

def get_agent_overrides(agent_id: str) -> dict:
    try:
        with _conn() as c:
            r = c.execute("SELECT overrides FROM agent_overrides WHERE agent_id=?", (agent_id,)).fetchone()
    except sqlite3.OperationalError:   # DB not initialised (scripts that only read YAML)
        return {}
    return json.loads(r["overrides"]) if r else {}


def set_agent_overrides(agent_id: str, overrides: dict) -> None:
    with _conn() as c:
        if overrides:
            c.execute("INSERT OR REPLACE INTO agent_overrides (agent_id, overrides, updated_at) "
                      "VALUES (?,?,?)", (agent_id, json.dumps(overrides, ensure_ascii=False), time.time()))
        else:
            c.execute("DELETE FROM agent_overrides WHERE agent_id=?", (agent_id,))
