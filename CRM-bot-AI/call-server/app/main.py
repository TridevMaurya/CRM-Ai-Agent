"""Gyan TV CRM — AI calling agents (Genie) with a dashboard.

Routes
  GET  /                         dashboard: agents, their WebSocket URLs, recent calls
  GET  /calls/{call_id}          one call: transcript, analysis, cost
  GET  /test/{agent_id}          talk to an agent from the browser (mic)
  GET  /api/agents               agents + stats
  GET  /api/calls                recent calls (?agent_id=)
  GET  /api/calls/{call_id}      call detail
  GET  /api/calls/{call_id}/recording   the Exotel recording (mp3), streamed through
  POST /api/context              store per-call context for a number (your own dialer)
  POST /api/dial                 dial a number via Exotel and store its context
  GET  /api/campaign             sheet sync + dialer status, queue counts
  POST /api/campaign/sync        sync the content + user sheets now
  POST /api/campaign/dialer      {"enabled": true|false, "budget"?: n, "window"?: "10:00-21:00"} — start / pause auto-dialing
  GET  /api/contacts             campaign queue (?status=)
  POST /api/contacts/{id}/dial   dial one queued contact right now
  GET  /admin                    admin panel (ADMIN_PASSWORD): agent settings, voice preview
  *    /api/admin/...             see app/admin.py
  GET  /health
  WS   /ws/{agent_id}            Exotel Voicebot applet endpoint (phone calls)
  WS   /ws/browser/{agent_id}    browser test calls (protobuf frames + JSON captions)
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.exotel import ExotelFrameSerializer
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pydantic import BaseModel, Field

from . import agents as agents_mod
from . import admin, analysis, campaign, exotel, store
from .pipeline import INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE, CallSession, make_vad, pregen_greeting, warm_clips

from .paths import PKG_ROOT as BASE_DIR, STATIC_DIR, ENV_FILE, SA_FILE  # noqa: E402

if ENV_FILE:
    load_dotenv(ENV_FILE)

# Hosted deploys can't ship the gitignored service-account file; accept the JSON
# via env and materialise it on boot (Google's client wants a file path). Locally
# the file is found next to the code / project (see paths.py).
_sa_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
if _sa_json and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    _sa_path = BASE_DIR / "gcp-service-account.json"
    _sa_path.write_text(_sa_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_sa_path)
elif not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and SA_FILE:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(SA_FILE)
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8080"))
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
DASHBOARD_TOKEN = os.getenv("DASHBOARD_TOKEN", "")

store.init()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Sheet sync + auto-dialer. Dialing itself stays off until switched on from the
    # dashboard (or DIALER_ENABLED=1).
    loop = asyncio.create_task(campaign.run_forever())
    warm = asyncio.create_task(warm_clips(list(agents_mod.load_agents().values())))
    yield
    warm.cancel()
    loop.cancel()


app = FastAPI(title="Gyan TV CRM — Genie", lifespan=_lifespan)
app.include_router(admin.router)


# ----- optional shared-token protection for pages + API (never for /ws) -----

@app.middleware("http")
async def _auth(request: Request, call_next):
    if DASHBOARD_TOKEN and not request.url.path.startswith(("/health", "/ws")):
        token = request.query_params.get("token") or request.cookies.get("dash_token")
        if token != DASHBOARD_TOKEN:
            return JSONResponse({"detail": "add ?token=<DASHBOARD_TOKEN> to the URL once"}, 401)
        if request.query_params.get("token"):
            resp = RedirectResponse(request.url.path)
            resp.set_cookie("dash_token", token, httponly=True, max_age=30 * 86400)
            return resp
    return await call_next(request)


# ----- pages -----

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/calls/{call_id}")
async def call_page(call_id: str):
    return FileResponse(STATIC_DIR / "call.html")


@app.get("/admin")
async def admin_page():
    return FileResponse(STATIC_DIR / "admin.html")


@app.get("/test/{agent_id}")
async def test_page(agent_id: str):
    return FileResponse(STATIC_DIR / "test.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def _no_stale_static(request: Request, call_next):
    # Dashboard files change with every deploy; a cached style.css from the previous
    # design once rendered the new page unstyled. Browsers must revalidate.
    resp = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path in ("/", "/admin") \
            or request.url.path.startswith(("/calls/", "/test/")):
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
    return resp


@app.get("/health")
async def health():
    return {"ok": True, "agents": list(agents_mod.load_agents())}


# ----- API -----

def _ws_base(request: Request) -> str:
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
    scheme = "wss" if request.url.scheme == "https" else "ws"
    return f"{scheme}://{request.url.netloc}"


@app.get("/api/agents")
async def api_agents(request: Request):
    stats = store.agent_stats()
    base = _ws_base(request)
    out = []
    for a in agents_mod.load_agents().values():
        out.append({
            "id": a["id"], "name": a["name"], "description": a.get("description", ""),
            "language": a["language"], "voice": a["voice"], "file": a["file"],
            "ws_url": f"{base}/ws/{a['id']}",
            "test_url": f"/test/{a['id']}",
            "dial_ready": exotel.configured() and bool((a.get("exotel") or {}).get("app_id")),
            "analysis_fields": list((a.get("analysis") or {}).keys()),
            "cohorts": a["cohorts"], "fillers": a["fillers"],
            "defaults": {**(a.get("defaults") or {}), **store.get_agent_default(a["id"])},
            "stats": stats.get(a["id"], {"n": 0, "live": 0, "avg_secs": None, "cost": None}),
        })
    return out


def _since(range_: str | None) -> float | None:
    from datetime import datetime, timedelta, timezone
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist)
    if range_ == "today":
        return now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    if range_ == "7d":
        return (now - timedelta(days=7)).timestamp()
    if range_ == "30d":
        return (now - timedelta(days=30)).timestamp()
    return None


@app.get("/api/calls")
async def api_calls(agent_id: str | None = None, sentiment: str | None = None,
                    transport: str | None = None, limit: int = 100, page: int = 1,
                    range: str | None = None, cohort: str | None = None, theme: str | None = None):
    """sentiment: positive | neutral | negative | none. range: today | 7d | 30d | all."""
    limit = min(limit, 500)
    return store.list_calls(limit=limit, agent_id=agent_id, sentiment=sentiment, transport=transport,
                            offset=(max(page, 1) - 1) * limit, since=_since(range), cohort=cohort,
                            theme=theme)


@app.get("/api/calls/insights")
async def api_calls_insights(agent_id: str | None = None, range: str | None = None,
                             cohort: str | None = None, transport: str | None = "exotel",
                             theme: str | None = None):
    return store.insights(agent_id, _since(range), cohort, transport or None, theme)


@app.get("/api/calls/summary")
async def api_calls_summary(transport: str | None = None):
    import time as _t
    from datetime import datetime, timedelta, timezone
    ist = timezone(timedelta(hours=5, minutes=30))
    midnight = datetime.now(ist).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    return {"counts": store.sentiment_counts(transport), "total": store.count_calls(transport=transport),
            "today": store.kpis(midnight), "all": store.kpis(0)}


EXPORT_COLS = ["started", "agent", "phone", "language", "cohort", "theme", "video_title", "status", "end_reason",
               "duration_secs", "cost_inr", "latency_p50", "sentiment", "summary", "primary_reason",
               "reason_detail", "willing_price", "refund_or_money_expectation", "liked", "praise", "complaints",
               "wants_more_of", "missing_content", "other_apps", "wanted_video", "proof_needed", "why_gyantv", "other_apps_gap", "one_improvement", "do_not_call",
               "transcript"]


@app.get("/api/calls/export.xlsx")
async def api_calls_export(agent_id: str | None = None, sentiment: str | None = None,
                           transport: str | None = None, range: str | None = None, cohort: str | None = None,
                           theme: str | None = None):
    """Every matching call as an Excel sheet (analysis fields + transcript)."""
    import io
    from datetime import datetime, timedelta, timezone

    from openpyxl import Workbook

    ist = timezone(timedelta(hours=5, minutes=30))
    wb = Workbook(); ws = wb.active; ws.title = "calls"
    ws.append(EXPORT_COLS)
    for c in store.list_calls(limit=100000, agent_id=agent_id, sentiment=sentiment, transport=transport,
                              since=_since(range), cohort=cohort, theme=theme):
        full = store.get_call(c["id"]) or c
        a = full.get("analysis") or {}; p = full.get("params") or {}
        row = {
            "started": datetime.fromtimestamp(full["started_at"], ist).strftime("%Y-%m-%d %H:%M"),
            "agent": full["agent_id"], "phone": full.get("from_number", ""), "language": full.get("language", ""),
            "cohort": p.get("cohort", ""), "theme": p.get("theme", ""), "video_title": p.get("video_title", ""),
            "status": full["status"],
            "end_reason": full.get("error", ""), "duration_secs": round(full.get("duration_secs") or 0),
            "cost_inr": full.get("cost_inr"), "latency_p50": full.get("latency_p50"),
            "transcript": "\n".join(f"{'Genie' if t['role'] == 'bot' else 'User'}: {t['text']}"
                                    for t in full.get("turns", [])),
        }
        for k in EXPORT_COLS:
            if k not in row:
                v = a.get(k, "")
                row[k] = "; ".join(map(str, v)) if isinstance(v, list) else ("" if v is None else v)
        ws.append([row[k] for k in EXPORT_COLS])
    ws.freeze_panes = "A2"
    for col, width in zip("ABCDEFGHIJKLM", (16, 22, 13, 8, 16, 30, 8, 18, 8, 8, 8, 10, 60)):
        ws.column_dimensions[col].width = width
    buf = io.BytesIO(); wb.save(buf)
    name = f"genie-calls-{agent_id or 'all'}-{sentiment or 'all'}.xlsx"
    return Response(buf.getvalue(),
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/calls/{call_id}")
async def api_call(call_id: str):
    call = store.get_call(call_id)
    if not call:
        raise HTTPException(404, "call not found")
    return call


@app.get("/api/calls/{call_id}/recording")
async def api_call_recording(call_id: str, request: Request):
    """Exotel records every call; play it here without storing a copy. Open to anyone
    with the dashboard link (Tridev's call: the team reviews recordings together)."""
    call = store.get_call(call_id)
    if not call:
        raise HTTPException(404, "call not found")
    if call["transport"] != "exotel":
        raise HTTPException(404, "browser test calls are not recorded")
    if not exotel.configured():
        raise HTTPException(400, "EXOTEL_* env vars not set")
    try:
        details = await asyncio.to_thread(exotel.call_details, call_id)
        url = details.get("RecordingUrl") or ""
        if not url:
            raise HTTPException(404, "recording not ready yet — Exotel usually needs a minute after the call")
        audio = await asyncio.to_thread(exotel.fetch_recording, url)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Exotel: {e}") from None
    return Response(audio, media_type="audio/mpeg",
                    headers={"Cache-Control": "private, max-age=3600"})


class ContextBody(BaseModel):
    agent_id: str
    number: str = ""                            # blank = set the agent's default context
    params: dict = Field(default_factory=dict)  # user_name, video_title, language, …


@app.post("/api/context", status_code=201)
async def api_context(body: ContextBody):
    """Store what Genie should know before a call. With a number: for that number's
    next call (your dialer / an Exotel campaign places it). Without a number: the
    agent's default context, used for any call with no number-specific context —
    e.g. someone calling the ExoPhone themselves."""
    if not agents_mod.get_agent(body.agent_id):
        raise HTTPException(404, "agent not found")
    if not body.number.strip():
        store.set_agent_default(body.agent_id, body.params)
        return {"ok": True, "default_for": body.agent_id, "params": body.params}
    if len(store.normalize_number(body.number)) < 10:
        raise HTTPException(400, "number must have 10 digits")
    store.put_pending(body.number, body.agent_id, body.params)
    return {"ok": True, "number": store.normalize_number(body.number)}


@app.post("/api/dial")
async def api_dial(body: ContextBody):
    agent = agents_mod.get_agent(body.agent_id)
    if not agent:
        raise HTTPException(404, "agent not found")
    exo = agent.get("exotel") or {}
    if not exotel.configured():
        raise HTTPException(400, "EXOTEL_* env vars not set")
    if not exo.get("app_id") or not exo.get("caller_id"):
        raise HTTPException(400, f"set exotel.app_id and exotel.caller_id in agents/{agent['file']}")
    if len(store.normalize_number(body.number)) < 10:
        raise HTTPException(400, "number must have 10 digits")
    store.put_pending(body.number, body.agent_id, body.params)
    asyncio.create_task(pregen_greeting(body.number, agent, body.params))
    ok, result = exotel.dial(body.number, exo["app_id"], exo["caller_id"])
    if not ok:
        raise HTTPException(502, result)
    return {"ok": True, "call_sid": result}


# ----- campaign (sheets → queue → auto-dialer) -----

@app.get("/api/campaign")
async def api_campaign():
    return campaign.status()


@app.post("/api/campaign/sync")
async def api_campaign_sync():
    if not (campaign.USER_SHEET or campaign.CONTENT_SHEET):
        raise HTTPException(400, "set USER_SHEET and CONTENT_SHEET in .env")
    return await campaign.sync()


class DialerBody(BaseModel):
    enabled: bool
    budget: int | None = None     # dials before the dialer pauses itself; omit = unlimited
    window: str | None = None     # daily IST window "HH:MM-HH:MM"; omit = keep current
    focus: dict | None = None     # {"video_category": "AI", "cohorts": [...], "themes_first": []}; omit = keep
    clear_focus: bool = False     # back to the env defaults


@app.post("/api/campaign/dialer")
async def api_campaign_dialer(body: DialerBody):
    if body.window:
        try:
            campaign.set_call_window(body.window)
        except ValueError as e:
            raise HTTPException(400, f"bad window: {e}")
    if body.clear_focus:
        campaign.set_dial_focus(None)
    elif body.focus is not None:
        campaign.set_dial_focus(body.focus)
    campaign.set_dialer_enabled(body.enabled, body.budget)
    return campaign.status()


@app.get("/api/contacts")
async def api_contacts(status: str | None = None, limit: int = 200):
    return store.list_contacts(limit=min(limit, 1000), status=status)


@app.post("/api/contacts/{contact_id}/dial")
async def api_contact_dial(contact_id: int):
    contact = store.get_contact(contact_id)
    if not contact:
        raise HTTPException(404, "contact not found")
    if contact["status"] in ("dialing", "in_call"):
        raise HTTPException(409, f"already {contact['status']}")
    if contact["status"] == "dnc":
        raise HTTPException(409, "this user asked not to be called")
    ok, result = await campaign.dial_contact(contact)
    if not ok:
        raise HTTPException(502, result)
    return {"ok": True, "call_sid": result}


# ----- WebSocket: Exotel (phone) -----

def _parse_custom_params(raw) -> dict:
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, str) and raw.strip():
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except ValueError:
            return dict(p.split("=", 1) for p in raw.split(",") if "=" in p)
    return {}


@app.websocket("/ws/{agent_id}")
async def ws_exotel(websocket: WebSocket, agent_id: str):
    await websocket.accept()
    try:
        transport_type, call_data = await parse_telephony_websocket(websocket)
    except Exception as e:
        logger.warning(f"[ws] could not parse start event: {e}")
        await websocket.close(code=1008)
        return
    if transport_type != "exotel":
        logger.warning(f"[ws] expected exotel stream, got {transport_type!r}")
        await websocket.close(code=1008)
        return

    agent = agents_mod.get_agent(agent_id)
    if not agent:
        logger.warning(f"[ws] unknown agent {agent_id!r}")
        await websocket.close(code=1008)
        return

    # Per-call context, lowest to highest precedence: agent YAML defaults → default
    # context saved on the dashboard → context stored for this number by /api/dial,
    # /api/context or the campaign dialer → Exotel custom_parameters.
    # Number-specific context means we dialed them (outbound). None means they called
    # the ExoPhone (inbound): if the number is in the campaign queue (usually someone
    # returning our missed call) they get that contact's agent and context.
    custom = _parse_custom_params(call_data.get("custom_parameters"))
    pending = store.pop_pending(call_data.get("from", "")) or store.pop_pending(call_data.get("to", ""))
    known: dict = {}
    route_agent = ""
    if pending:
        route_agent, known = pending
    elif contact := campaign.context_for_incoming(call_data.get("from", "")):
        route_agent, known = contact["agent_id"], {**contact["params"], "contact_id": contact["id"]}
    if route_agent and route_agent != agent_id and agents_mod.get_agent(route_agent):
        agent, agent_id = agents_mod.get_agent(route_agent), route_agent
    # Campaign contacts carry their own full context — the demo defaults (a made-up
    # name / video) must not leak into a real user's call.
    fallback = ({} if ("contact_id" in known or "cohort" in known or "video_title" in known)
                else {**(agent.get("defaults") or {}), **store.get_agent_default(agent_id)})
    params = {**fallback, **known, **custom}
    params["direction"] = "outbound" if pending else "inbound"
    params["_number"] = call_data.get("from", "")   # for the pre-rendered greeting

    call_id = call_data.get("call_id") or f"exo_{uuid.uuid4().hex[:12]}"
    campaign.call_started(params.get("contact_id"), call_id)
    lang = agents_mod.resolve_language(agent, params)
    store.create_call(call_id, agent_id, "exotel", from_number=call_data.get("from", ""),
                      to_number=call_data.get("to", ""), language=lang, params=params)
    logger.info(f"[{call_id}] exotel stream {call_data.get('stream_id')} from={call_data.get('from')} "
                f"to={call_data.get('to')} params={params}")

    vad = make_vad()
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True, audio_out_enabled=True, add_wav_header=False,
            audio_in_sample_rate=INPUT_SAMPLE_RATE, audio_out_sample_rate=OUTPUT_SAMPLE_RATE,
            vad_analyzer=vad,
            serializer=ExotelFrameSerializer(stream_sid=call_data["stream_id"],
                                             call_sid=call_data.get("call_id")),
        ),
    )
    await _run_and_analyse(CallSession(call_id=call_id, agent=agent, params=params,
                                       transport=transport, vad_analyzer=vad))


# ----- WebSocket: browser test call -----

@app.websocket("/ws/browser/{agent_id}")
async def ws_browser(websocket: WebSocket, agent_id: str):
    await websocket.accept()
    try:
        config = await websocket.receive_json()   # first message: {params: {...}}
    except WebSocketDisconnect:
        return
    agent = agents_mod.get_agent(agent_id)
    if not agent:
        await websocket.close(code=1008, reason="unknown agent")
        return
    given = {k: v for k, v in (config.get("params") or {}).items() if v not in (None, "")}
    params = {**(agent.get("defaults") or {}), **store.get_agent_default(agent_id), **given}
    if str(params.get("pregen", "")) == "1":       # test the pre-rendered opening path
        params["_number"] = "0000000000"
        await pregen_greeting(params["_number"], agent, params)
    call_id = f"web_{uuid.uuid4().hex[:12]}"
    lang = agents_mod.resolve_language(agent, params)
    store.create_call(call_id, agent_id, "browser", language=lang, params=params)

    vad = make_vad()
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True, audio_out_enabled=True, add_wav_header=False,
            audio_in_sample_rate=INPUT_SAMPLE_RATE, audio_out_sample_rate=OUTPUT_SAMPLE_RATE,
            vad_analyzer=vad, serializer=ProtobufFrameSerializer(),
        ),
    )
    await _run_and_analyse(CallSession(call_id=call_id, agent=agent, params=params,
                                       transport=transport, vad_analyzer=vad, caption_ws=websocket))


async def _run_and_analyse(session: CallSession):
    try:
        await session.run()
    except Exception as e:  # noqa: BLE001
        logger.exception(f"[{session.call_id}] call error: {e}")
    call = store.get_call(session.call_id)
    if call and call["turns"]:
        result = await analysis.analyse_call(session.agent, call["turns"])
        if result:
            store.set_analysis(session.call_id, result)
            logger.info(f"[{session.call_id}] analysis: {json.dumps(result, ensure_ascii=False)[:300]}")
            call["analysis"] = result
    if call:
        await campaign.call_finished(call)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
