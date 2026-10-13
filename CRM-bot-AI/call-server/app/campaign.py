"""Campaign: sheets in, calls out, results back.

    content sheet ─┐                         ┌─> Exotel dials ─> /ws/{agent} (Genie)
                   ├─ sync every N min ─> contacts queue ─ dialer loop
    user sheet ────┘   (SQLite)                               │
                                                              └─> call_log tab in the user sheet

The user sheet (analyst-owned, auto-refreshed from BigQuery) has one row per user
with a `cohort`. Each agent YAML lists the cohorts it handles, so the cohort picks
the agent. The dialer only runs inside the calling window, keeps at most
MAX_CONCURRENT_CALLS in flight, retries no-answers, and never calls the same
number twice within RECALL_AFTER_DAYS (or ever again after "don't call me").

Incoming calls: when someone rings the ExoPhone, main.py looks the number up here
and gives Genie that contact's agent + context (they're usually returning our call).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from loguru import logger

from .themes import norm_theme, themes_of
from . import agents as agents_mod
from . import exotel, sheets, store

# Settings below are read at import, before main.py loads .env — so load it here.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

IST = timezone(timedelta(hours=5, minutes=30))


def _env(key: str, default: str) -> str:
    return (os.getenv(key) or default).strip()


USER_SHEET = sheets.sheet_id_from(_env("USER_SHEET", ""))
# Tabs to read, comma-separated; blank = every tab except the call log.
USER_TABS = [t.strip() for t in _env("USER_SHEET_TABS", _env("USER_SHEET_TAB", "")).split(",") if t.strip()]
CALL_LOG_TAB = _env("CALL_LOG_TAB", "call_log")
CONTENT_SHEET = sheets.sheet_id_from(_env("CONTENT_SHEET", ""))
CONTENT_TAB = _env("CONTENT_SHEET_TAB", "")          # blank = first tab
SCRIPT_TABS = [t.strip() for t in _env("CONTENT_SCRIPT_TABS", "Hindi,Telugu,Tamil,Kannada,Malayalam").split(",") if t.strip()]

SYNC_EVERY_SECS = float(_env("SYNC_EVERY_MINS", "15")) * 60
CALL_WINDOW = _env("CALL_WINDOW", "10:00-19:00")      # IST
MAX_CONCURRENT_CALLS = int(_env("MAX_CONCURRENT_CALLS", "2"))   # connected conversations
# Ringing calls beyond that: ~half of dials are never answered, so a few extra
# ringing keeps all conversation slots busy (Exotel voicebot only counts connected).
MAX_RINGING = int(_env("MAX_RINGING", "3"))
DIAL_GAP_SECS = float(_env("DIAL_GAP_SECS", "20"))
MAX_ATTEMPTS = int(_env("MAX_ATTEMPTS", "3"))
RETRY_AFTER_SECS = float(_env("RETRY_AFTER_MINS", "180")) * 60
NO_ANSWER_AFTER_SECS = float(_env("NO_ANSWER_AFTER_SECS", "120"))
RECALL_AFTER_SECS = float(_env("RECALL_AFTER_DAYS", "30")) * 86400
SUMMARY_MODEL = _env("SUMMARY_MODEL", "gemini-3.1-flash-lite")
SUMMARIES_PER_SYNC = int(_env("SUMMARIES_PER_SYNC", "250"))   # ~2,500 Hindi scripts → ~10 syncs
# Only dial these languages (codes: hin, tel, tam, kan, mal, mar); blank = all.
DIAL_LANGUAGES = tuple(l.strip() for l in _env("DIAL_LANGUAGES", "").split(",") if l.strip())
# Cohorts to dial, in priority order (blank = all: SUB > TRIAL > DAU > LIGHT).
DIAL_COHORTS = tuple(c.strip().upper() for c in _env("DIAL_COHORTS", "").split(",") if c.strip())
# From NIGHT_FROM (IST) the dialer switches to DIAL_COHORTS_NIGHT — e.g. trial-cancelled
# all day, the content-research cohorts in the evening.
DIAL_COHORTS_NIGHT = tuple(c.strip().upper() for c in _env("DIAL_COHORTS_NIGHT", "").split(",") if c.strip())
NIGHT_FROM = _env("NIGHT_FROM", "19:00")


def dial_focus() -> dict:
    """A temporary targeting override set from the API, e.g. {"video_category": "AI",
    "cohorts": [...], "themes_first": []}: only contacts whose last video is in that
    category, in that cohort order. Empty = the env defaults."""
    raw = store.get_setting("dial_focus", "")
    try:
        return json.loads(raw) if raw else {}
    except ValueError:
        return {}


def set_dial_focus(focus: dict | None) -> None:
    store.set_setting("dial_focus", json.dumps(focus) if focus else "")


def dial_cohorts_now(now: datetime | None = None) -> tuple[str, ...]:
    focus = dial_focus()
    if focus.get("cohorts"):
        return tuple(str(c).upper() for c in focus["cohorts"])
    if not DIAL_COHORTS_NIGHT:
        return DIAL_COHORTS
    now = now or datetime.now(IST)
    night = datetime.strptime(NIGHT_FROM, "%H:%M").time()
    return DIAL_COHORTS_NIGHT if now.time() >= night else DIAL_COHORTS
# Themes dialed before everything else (any cohort): Saloni wants UPSC/PSC volume.
DIAL_THEMES_FIRST = tuple(t.strip() for t in _env("DIAL_THEMES_FIRST", "UPSC,PSC").split(",") if t.strip())

CALL_LOG_HEADER = [
    "logged_at", "call_id", "direction", "phone", "user_id", "cohort", "agent", "language",
    "status", "duration_secs", "video_title", "sentiment", "do_not_call", "summary", "analysis_json",
]

state = {"last_sync_at": 0.0, "last_sync": {}, "last_dial_at": 0.0, "last_error": ""}


# ----- settings -----

def dialer_enabled() -> bool:
    return store.get_setting("dialer_enabled", _env("DIALER_ENABLED", "0")) == "1"


def set_dialer_enabled(on: bool, budget: int | None = None) -> None:
    store.set_setting("dialer_enabled", "1" if on else "0")
    if on:
        # Starting: a number = dials before auto-pause; none = unlimited (a leftover 0
        # from the previous run must not pause the dialer on its first tick).
        store.set_setting("dial_budget", str(max(0, int(budget))) if budget else "")


def dial_budget() -> int | None:
    """Dials left before the dialer pauses itself; None = unlimited."""
    v = store.get_setting("dial_budget", "")
    return int(v) if v.strip().isdigit() else None


def call_window() -> str:
    """Daily IST dialing window "HH:MM-HH:MM"; settable at runtime, env is the default."""
    return store.get_setting("call_window", "") or CALL_WINDOW


def set_call_window(window: str) -> None:
    start, end = (datetime.strptime(t.strip(), "%H:%M").time() for t in window.split("-"))
    if start >= end:
        raise ValueError("window start must be before end")
    store.set_setting("call_window", window.strip())


def in_call_window(now: datetime | None = None) -> bool:
    now = now or datetime.now(IST)
    start, end = (datetime.strptime(t.strip(), "%H:%M").time() for t in call_window().split("-"))
    return start <= now.time() < end


def status() -> dict:
    return {
        "dialer_enabled": dialer_enabled(),
        "dial_budget": dial_budget(),
        "in_call_window": in_call_window(),
        "call_window_ist": call_window(),
        "max_concurrent": MAX_CONCURRENT_CALLS,
        "max_ringing": MAX_RINGING,
        "dial_themes_first": list(DIAL_THEMES_FIRST),
        "dial_focus": dial_focus(),
        "in_call": store.count_in_call(),
        "dial_languages": list(DIAL_LANGUAGES) or "all",
        "dial_cohorts": list(dial_cohorts_now()) or "all",
        "dial_cohorts_day": list(DIAL_COHORTS) or "all",
        "dial_cohorts_night": list(DIAL_COHORTS_NIGHT),
        "night_from": NIGHT_FROM if DIAL_COHORTS_NIGHT else "",
        "in_flight": store.count_in_flight(),
        "user_sheet": bool(USER_SHEET), "content_sheet": bool(CONTENT_SHEET),
        "service_account": sheets.service_account_email(),
        "content_items": store.content_count(),
        "last_sync_at": state["last_sync_at"], "last_sync": state["last_sync"],
        "last_error": state["last_error"],
        "queue": store.queue_summary(),
    }


# ----- sync -----

def _first(rec: dict, *keys: str) -> str:
    for k in keys:
        if rec.get(k):
            return rec[k]
    return ""


def _script_texts() -> dict[str, str]:
    """content_id → script (or description) from the per-language production tabs
    ("Hindi", "Telugu", …: columns Content Series ID / Video-Title / Description /
    Script). The "All Videos" tab has titles and categories but no text, and the
    10-second summary Genie uses to ask specific questions is made from this."""
    out: dict[str, str] = {}
    if not SCRIPT_TABS:
        return out
    try:
        present = set(sheets.tab_names(CONTENT_SHEET))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[campaign] content tabs: {e}")
        return out
    for tab in SCRIPT_TABS:
        if tab not in present:
            continue
        try:
            for rec in sheets.read_records(CONTENT_SHEET, tab, ("content_series_id", "content_id")):  # the row above the header has a "SCRIPT" group label
                cid = _first(rec, "content_series_id", "content_id")
                text = _first(rec, "script", "video_script", "transcript") or rec.get("description", "")
                if cid and len(text) > 40:
                    out[cid] = text
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[campaign] script tab {tab!r}: {e}")
    return out


def _sync_content() -> int:
    tab = CONTENT_TAB or sheets.tab_names(CONTENT_SHEET)[0]
    scripts = _script_texts()
    rows = []
    for rec in sheets.read_records(CONTENT_SHEET, tab, "content_id"):
        cid, title = rec.get("content_id", ""), _first(rec, "video_title", "title")
        if not cid or not title:
            continue
        rows.append({
            "content_id": cid, "title": title, "category": norm_theme(rec.get("category", "")),
            "language": rec.get("language", ""),
            "script": _first(rec, "script", "video_script", "script_text", "transcript") or scripts.get(cid, ""),
            "summary": _first(rec, "summary", "video_summary", "short_summary"),
        })
    store.upsert_content(rows)
    logger.info(f"[campaign] content: {len(rows)} items, {sum(1 for r in rows if r['script'])} with a script")
    return len(rows)


# Cohort labels as they appear in the sheets → our cohort keys. The analyst's
# BigQuery export uses "4-9 videos" / "10+ videos" and cancellation_type trial / subs.
COHORT_ALIASES = {
    "light_3_4": "LIGHT_3_4", "3-4 videos": "LIGHT_3_4", "4-9 videos": "LIGHT_3_4",
    "dau_10plus": "DAU_10PLUS", "10+ videos": "DAU_10PLUS",
    "trial_cancelled": "TRIAL_CANCELLED", "trial": "TRIAL_CANCELLED",
    "sub_cancelled": "SUB_CANCELLED", "subs": "SUB_CANCELLED", "subscription": "SUB_CANCELLED",
}
# One call per person: when a number is in several cohorts, the first here wins.
COHORT_PRIORITY = ["SUB_CANCELLED", "TRIAL_CANCELLED", "DAU_10PLUS", "LIGHT_3_4"]
CANCEL_COHORTS = ("TRIAL_CANCELLED", "SUB_CANCELLED")


def _cohort_of(rec: dict) -> str:
    raw = _first(rec, "cohort", "video_cohort", "cancellation_type").strip().lower()
    return COHORT_ALIASES.get(raw, raw.upper())


def _watched_ids(rec: dict) -> list[str]:
    """Most recent first. Accepts a comma list (watched_content_ids) or the export's
    video_1_course_id … video_N_course_id / latest_course_id, second_latest_course_id."""
    if rec.get("watched_content_ids"):
        return [c.strip() for c in rec["watched_content_ids"].split(",") if c.strip()]
    ids = [rec[f"video_{i}_course_id"] for i in range(1, 51) if rec.get(f"video_{i}_course_id")]
    ids += [rec[k] for k in ("latest_course_id", "second_latest_course_id") if rec.get(k)]
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _video_params(item: dict | None) -> dict:
    if not item:
        return {}
    return {"video_title": item["title"], "video_category": item["category"],
            "video_summary": item["summary"]}


def contact_params(rec: dict, cohort: str, content: dict[str, dict]) -> dict:
    """Everything a call needs, from one user-sheet row + the content catalogue."""
    watched = _watched_ids(rec)
    params = {
        "user_id": rec.get("user_id", ""),
        "user_name": rec.get("name", ""),
        "language": agents_mod.normalize_language(_first(rec, "language", "language_code")),
        "cohort": cohort,
        "total_videos_watched": _first(rec, "total_videos_watched", "videos_watched"),
        "avg_videos_per_day": rec.get("avg_videos_per_day", ""),
        "top_category": rec.get("top_category", ""),
        "subscription_status": rec.get("subscription_status", ""),
        "plan_price": rec.get("plan_price", ""),
        "cancel_at": _first(rec, "cancel_at", "cancellation_date"),
    }
    if cohort in CANCEL_COHORTS:
        video_id = _first(rec, "last_content_before_cancel", "latest_course_id")
    else:
        video_id = rec.get("last_watched_content_id", "") or (watched[0] if watched else "")
    params["video_id"] = video_id
    params.update(_video_params(content.get(video_id)))
    # "watched_videos" separates "watched, but we don't know the title" (no content
    # sheet row) from "watched nothing" — the prompts branch on it.
    params["watched_videos"] = "yes" if (video_id or watched) else "no"
    titles = [content[w]["title"] for w in watched[:5] if w in content]
    if titles:
        params["watched_titles"] = " | ".join(titles)
    if not params.get("top_category"):
        cats = [content[w]["category"] for w in watched if w in content and content[w]["category"]]
        if cats:
            params["top_category"] = max(set(cats), key=cats.count)
    # Themes from EVERYTHING they watched (not just the last video): primary theme is
    # UPSC/PSC if any video was, else the most-watched category; `themes` keeps all.
    cats = [content[w]["category"] for w in watched if w in content and content[w]["category"]]
    for extra in (params.get("video_category"), params.get("top_category")):
        if extra:
            cats.append(extra)
    primary, all_themes = themes_of(cats)
    params["theme"] = primary
    params["themes"] = all_themes
    return {k: v for k, v in params.items() if v not in ("", None, [])}


def _sync_users() -> dict:
    counts = {"rows": 0, "invalid": 0, "duplicate": 0}
    agent_of = {c: (agents_mod.agent_for_cohort(c) or {}).get("id") for c in COHORT_PRIORITY}
    tabs = USER_TABS or [t for t in sheets.tab_names(USER_SHEET) if t != CALL_LOG_TAB]
    best: dict[str, tuple[int, str, dict]] = {}   # phone → (priority, cohort, row)
    for tab in tabs:
        for rec in sheets.read_records(USER_SHEET, tab, ("phone", "mobile_no")):
            counts["rows"] += 1
            phone = store.normalize_number(_first(rec, "phone", "mobile_no"))
            cohort = _cohort_of(rec)
            if len(phone) != 10 or phone[0] not in "6789" or not agent_of.get(cohort):
                counts["invalid"] += 1
                continue
            rank = COHORT_PRIORITY.index(cohort)
            if phone in best:
                counts["duplicate"] += 1
                if best[phone][0] <= rank:
                    continue
            best[phone] = (rank, cohort, rec)
    content = store.all_content()
    rows = [(phone, cohort, agent_of[cohort], contact_params(rec, cohort, content))
            for phone, (_, cohort, rec) in best.items()]
    counts.update(store.upsert_contacts(rows, RECALL_AFTER_SECS))
    counts["by_cohort"] = {c: sum(1 for r in rows if r[1] == c) for c in COHORT_PRIORITY}
    counts["with_title"] = sum(1 for r in rows if r[3].get("video_title"))
    return counts


async def _summarise_scripts() -> int:
    """One ~10-second spoken summary per video script (once; cached in SQLite)."""
    todo = store.content_needing_summary(SUMMARIES_PER_SYNC)
    if not todo:
        return 0
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    done = 0
    for item in todo:
        prompt = ("A phone caller (Genie) will ask a viewer specific questions about this video. "
                  "Write 2-3 short sentences, in the same language and script as the SCRIPT, that say: "
                  "what the video shows or claims, and the 2-3 concrete things it tells the viewer to do "
                  "or remember (the steps, the trick, the tip, the numbers). Plain text only, no preamble, "
                  "no bullet marks.\n\nTITLE: "
                  f"{item['title']}\n\nSCRIPT:\n{item['script'][:12000]}")
        try:
            resp = await client.aio.models.generate_content(
                model=SUMMARY_MODEL, contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3))
            if (resp.text or "").strip():
                store.set_content_summary(item["content_id"], resp.text.strip())
                done += 1
        except Exception as e:
            logger.warning(f"[campaign] summary for {item['content_id']} failed: {e}")
    return done


_summary_task: asyncio.Task | None = None


async def _summarise_all() -> None:
    total = 0
    while True:
        n = await _summarise_scripts()
        total += n
        if n == 0:
            break
        await asyncio.sleep(1)
    if total:
        logger.info(f"[campaign] summaries: {total} generated")


def _kick_summaries() -> None:
    """Generate missing video summaries in a background task (one run at a time).
    It used to run inside the sync, inside the dial tick: 250 Gemini calls in a row
    stalled dialing for ~8 minutes every sync."""
    global _summary_task
    if _summary_task is None or _summary_task.done():
        _summary_task = asyncio.create_task(_summarise_all())


async def sync() -> dict:
    result: dict = {"at": time.time()}
    try:
        if CONTENT_SHEET:
            result["content"] = await asyncio.to_thread(_sync_content)
            _kick_summaries()            # in the background — must not hold up the dialer
            result["summaries"] = "background"
        if USER_SHEET:
            result["users"] = await asyncio.to_thread(_sync_users)
        state["last_error"] = ""
    except Exception as e:  # noqa: BLE001 — a bad sheet must not stop the dialer
        state["last_error"] = f"sync: {e}"
        result["error"] = str(e)
        logger.warning(f"[campaign] sync failed: {e}")
    state["last_sync_at"], state["last_sync"] = time.time(), result
    logger.info(f"[campaign] sync {result}")
    return result


# ----- dialing -----

def _retry_or_give_up(contact: dict, final_status: str, note: str) -> None:
    if contact["attempts"] < MAX_ATTEMPTS:
        store.update_contact(contact["id"], status="queued", note=note,
                             next_attempt_at=time.time() + RETRY_AFTER_SECS)
    else:
        store.update_contact(contact["id"], status=final_status, note=note)


async def dial_contact(contact: dict) -> tuple[bool, str]:
    agent = agents_mod.get_agent(contact["agent_id"])
    exo = (agent or {}).get("exotel") or {}
    if not agent or not exo.get("app_id") or not exo.get("caller_id") or not exotel.configured():
        return False, "agent / Exotel not configured"
    params = {**contact["params"], "contact_id": contact["id"]}
    if not params.get("video_summary") and params.get("video_id"):
        # Summaries are generated after the contact was synced; pick up the latest one
        # at dial time instead of waiting for the next user-sheet sync.
        item = store.get_content(str(params["video_id"]))
        if item and item.get("summary"):
            params["video_summary"] = item["summary"]
    store.put_pending(contact["phone"], agent["id"], params)
    from .pipeline import pregen_greeting
    asyncio.create_task(pregen_greeting(contact["phone"], agent, params))
    store.update_contact(contact["id"], status="dialing", attempts=contact["attempts"] + 1,
                         last_dialed_at=time.time(), note="")
    ok, result = await asyncio.to_thread(exotel.dial, contact["phone"], exo["app_id"], exo["caller_id"])
    if not ok and _account_problem(result):
        # Exotel refused us, not the number (no balance, bad credentials): the contact
        # keeps its attempt, and the dialer pauses itself instead of failing the queue.
        store.update_contact(contact["id"], status="queued", attempts=contact["attempts"],
                             next_attempt_at=time.time(), note="")
        set_dialer_enabled(False)
        state["last_error"] = f"Dialer paused — Exotel: {result[:160]}"
        logger.error(f"[campaign] {state['last_error']}")
    elif not ok:
        _retry_or_give_up({**contact, "attempts": contact["attempts"] + 1}, "failed", result[:200])
    logger.info(f"[campaign] dial {contact['phone']} ({contact['cohort']}) → {ok} {result}")
    return ok, result


def _account_problem(result: str) -> bool:
    r = (result or "").lower()
    return ("insufficient balance" in r or "recharge" in r or "HTTP 401" in (result or "")
            or "authentication" in r)


async def _dial_next() -> None:
    focus = dial_focus()
    themes_first = tuple(focus["themes_first"]) if "themes_first" in focus else DIAL_THEMES_FIRST
    contact = store.next_due_contact(DIAL_LANGUAGES, dial_cohorts_now(), themes_first,
                                     video_category=focus.get("video_category") or None)
    if not contact:
        return
    if store.phone_is_dnc(contact["phone"]):
        store.update_contact(contact["id"], status="dnc", note="asked not to be called")
        return
    if store.recently_called(contact["phone"], RECALL_AFTER_SECS, exclude_id=contact["id"]):
        store.update_contact(contact["id"], status="skipped", note="called recently in another cohort")
        return
    budget = dial_budget()
    if budget is not None:
        if budget <= 0:
            set_dialer_enabled(False)
            logger.info("[campaign] dial budget used up — dialer paused")
            return
        store.set_setting("dial_budget", str(budget - 1))
    state["last_dial_at"] = time.time()
    await dial_contact(contact)


async def _tick() -> None:
    for c in store.stale_dialing(NO_ANSWER_AFTER_SECS):
        _retry_or_give_up(c, "no_answer", "no answer")
    if time.time() - state["last_sync_at"] >= SYNC_EVERY_SECS and (USER_SHEET or CONTENT_SHEET):
        await sync()
    # Fill every free slot this tick, not one per tick: with ~32% pickup and ~50 s
    # calls, one dial per 5 s tops out at ~3 conversations however many channels exist.
    while (dialer_enabled() and in_call_window()
           and store.count_in_call() < MAX_CONCURRENT_CALLS
           and store.count_in_flight() < MAX_CONCURRENT_CALLS + MAX_RINGING
           and time.time() - state["last_dial_at"] >= DIAL_GAP_SECS):
        before = store.count_in_flight()
        await _dial_next()
        if store.count_in_flight() == before:   # nothing dialable right now
            break
        await asyncio.sleep(DIAL_GAP_SECS)


def requeue_account_failures() -> int:
    """Contacts marked failed by an Exotel account problem (balance, auth) were never
    really dialed: put them back with their attempts restored."""
    n = 0
    for c in store.list_contacts(limit=100000, status="failed"):
        if _account_problem(c.get("note") or ""):
            store.update_contact(c["id"], status="queued", attempts=max(0, c["attempts"] - 1),
                                 next_attempt_at=time.time(), note="")
            n += 1
    if n:
        logger.info(f"[campaign] re-queued {n} contacts failed by an Exotel account problem")
    return n


async def run_forever() -> None:
    requeue_account_failures()
    logger.info(f"[campaign] loop started · user_sheet={bool(USER_SHEET)} content_sheet="
                f"{bool(CONTENT_SHEET)} window={call_window()} IST max={MAX_CONCURRENT_CALLS}")
    while True:
        try:
            await _tick()
        except Exception as e:  # noqa: BLE001
            state["last_error"] = str(e)
            logger.exception(f"[campaign] tick failed: {e}")
        await asyncio.sleep(5)


# ----- call lifecycle hooks (called from main.py) -----

def context_for_incoming(phone: str) -> dict | None:
    """The contact behind an incoming call, if we know the number."""
    # A DNC contact who calls us themselves still gets their own context — they chose
    # to call; we just never dial them.
    return store.latest_contact_for_phone(phone)


def call_started(contact_id, call_id: str) -> None:
    if contact_id:
        store.update_contact(int(contact_id), status="in_call", call_id=call_id)


async def call_finished(call: dict) -> None:
    """Move the contact on, and append the call to the sheet's call_log tab."""
    params, analysis = call.get("params") or {}, call.get("analysis") or {}
    user_turns = sum(1 for t in call.get("turns", []) if t["role"] == "user")
    dnc = str(analysis.get("do_not_call", "")).lower() == "true"
    contact_id = params.get("contact_id")
    if contact_id:
        contact = store.get_contact(int(contact_id))
        if contact:
            if dnc:
                store.update_contact(contact["id"], status="dnc", note="asked not to be called")
            elif user_turns == 0:
                _retry_or_give_up(contact, "no_answer", "picked up but never spoke")
            else:
                store.update_contact(contact["id"], status="done", note="")
    if call.get("transport") != "exotel" or not USER_SHEET:
        return
    row = [
        datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"), call["id"], params.get("direction", ""),
        store.normalize_number(call.get("from_number", "")), params.get("user_id", ""),
        params.get("cohort", ""), call["agent_id"], call.get("language", ""), call.get("status", ""),
        round(call.get("duration_secs") or 0), params.get("video_title", ""),
        analysis.get("sentiment", ""), "yes" if dnc else "", analysis.get("summary", ""),
        json.dumps(analysis, ensure_ascii=False),
    ]
    try:
        await asyncio.to_thread(sheets.ensure_tab, USER_SHEET, CALL_LOG_TAB, CALL_LOG_HEADER)
        await asyncio.to_thread(sheets.append_rows, USER_SHEET, CALL_LOG_TAB, [row])
    except Exception as e:  # noqa: BLE001
        state["last_error"] = f"call_log: {e}"
        logger.warning(f"[campaign] call_log append failed: {e}")
