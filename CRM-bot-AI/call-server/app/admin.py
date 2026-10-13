"""Admin panel API — view and change each agent's settings without a deploy.

Edits are stored as overrides over the agent YAML (store.agent_overrides) and
apply from the next call (agents are re-read per call). "Reset" drops them and the
YAML values come back.

Login: ADMIN_PASSWORD (env / Fly secret). Unset → the admin API is disabled.
The cookie holds an HMAC of the password, never the password itself.

Routes (all under /api/admin, all but login need the cookie)
  POST   /login                    {"password"}
  POST   /logout
  GET    /agents                   every agent: effective values, YAML values, overrides, schema
  PUT    /agents/{id}              {"values": {"tts.provider": "gemini_lite", ...}} → save
  DELETE /agents/{id}              reset to YAML
  POST   /preview                  {"provider","voice","style","speed","language","text"} → audio/wav
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import io
import os
import wave

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from . import agents as agents_mod
from . import store
from .gemini_tts import MODELS as GEMINI_TTS_MODELS
from .gemini_tts import synthesize as gemini_synthesize

router = APIRouter(prefix="/api/admin")
COOKIE = "genie_admin"

# Chirp3-HD / Gemini TTS prebuilt voices (same 30 names on both).
VOICES = [
    "Achernar", "Achird", "Algenib", "Algieba", "Alnilam", "Aoede", "Autonoe", "Callirrhoe",
    "Charon", "Despina", "Enceladus", "Erinome", "Fenrir", "Gacrux", "Iapetus", "Kore",
    "Laomedeia", "Leda", "Orus", "Puck", "Pulcherrima", "Rasalgethi", "Sadachbia",
    "Sadaltager", "Schedar", "Sulafat", "Umbriel", "Vindemiatrix", "Zephyr", "Zubenelgenubi",
]

# Every editable setting: dotted path into the agent YAML → how to validate it.
# (group, label, type, extra) — type: enum | bool | int | float | str | text
SCHEMA: dict[str, dict] = {
    "tts.provider": {"group": "Voice", "label": "TTS engine", "type": "enum",
                     "options": ["chirp", "eleven", "gemini_lite", "gemini_flash"],
                     "help": "chirp: ~0.3 s to first audio. eleven: ElevenLabs v4 Turbo (~0.5 s; needs the voice id below). "
                             "gemini_lite / gemini_flash: 3-5 s to first audio measured 8 Oct 2026."},
    "tts.voice_id": {"group": "Voice", "label": "ElevenLabs voice id", "type": "str",
                     "help": "Used only when TTS engine is eleven, e.g. RDWdsTU6N02BFftbIEAp."},
    "tts.model": {"group": "Voice", "label": "ElevenLabs model", "type": "str",
                  "help": "eleven_v4_turbo (fast, $11/1M chars) or eleven_v4 (richer, $22/1M)."},
    "voice": {"group": "Voice", "label": "Voice", "type": "enum", "options": VOICES},
    "tts.style": {"group": "Voice", "label": "Style (Gemini only)", "type": "str",
                  "help": "Delivery prompt, e.g. \"warm, friendly, natural conversational pace\"."},
    "speed": {"group": "Voice", "label": "Speed (Chirp only)", "type": "float", "min": 0.5, "max": 1.5},
    "language": {"group": "Conversation", "label": "Default language", "type": "enum",
                 "options": list(agents_mod.LANG_NAMES),
                 "help": "Used when the call / sheet row doesn't say."},
    "temperature": {"group": "Conversation", "label": "LLM temperature", "type": "float", "min": 0, "max": 1.5},
    "greeting_delay_secs": {"group": "Conversation", "label": "Pause before the opening (s, outbound)", "type": "float", "min": 0, "max": 3,
                            "help": "Wait this long after pickup before Genie speaks, so the first words are not lost."},
    "greeting_wait_secs": {"group": "Conversation", "label": "Wait for their 'hello' (s, outbound)", "type": "float", "min": 0, "max": 5,
                           "help": "If the user speaks within this time after pickup, Genie lets them finish and then opens; else she opens after the pause above."},
    "stt.provider": {"group": "Voice", "label": "Speech-to-text", "type": "enum", "options": ["sarvam", "eleven"],
                     "help": "sarvam = Sarvam saaras v3 (default). eleven = ElevenLabs Scribe v2 realtime (A/B; per-call param stt=eleven)."},
    "max_call_secs": {"group": "Conversation", "label": "Max call length (s)", "type": "int", "min": 60, "max": 300,
                      "help": "Hard cut at 5 minutes, always."},
    "silence_hangup_secs": {"group": "Conversation", "label": "Hang up after silence (s)", "type": "int",
                            "min": 10, "max": 300},
    "turn_end_secs": {"group": "Conversation", "label": "Wait after user stops (s)", "type": "float",
                      "min": 0.5, "max": 4,
                      "help": "Silence before Genie replies to a finished-sounding sentence."},
    "turn_hold_secs": {"group": "Conversation", "label": "Wait when sentence sounds unfinished (s)",
                       "type": "float", "min": 0.5, "max": 6,
                       "help": "Used when their words end mid-thought (\"matlab…\", \"aur…\", \"ki…\")."},
    "interruptions.enabled": {"group": "Barge-in", "label": "User can interrupt Genie", "type": "bool"},
    "interruptions.min_words": {"group": "Barge-in", "label": "Words needed to interrupt", "type": "int",
                                "min": 1, "max": 6},
    "fillers.enabled": {"group": "Turn-end filler", "label": "\"okay / achha\" when the user stops", "type": "bool"},
    "fillers.every": {"group": "Turn-end filler", "label": "Every Nth turn", "type": "int", "min": 1, "max": 5},
    "fillers.hums_only": {"group": "Turn-end filler", "label": "Only the recorded hums (no words)", "type": "bool"},
    "fillers.gain": {"group": "Turn-end filler", "label": "Volume (0–1)", "type": "float", "min": 0, "max": 1},
    "fillers.delay_secs": {"group": "Turn-end filler", "label": "Seconds after the user stops", "type": "float", "min": 0, "max": 5},
    "backchannel.enabled": {"group": "Backchannel", "label": "Hums while the user talks", "type": "bool"},
    "backchannel.min_gap": {"group": "Backchannel", "label": "Min gap (s)", "type": "float", "min": 0.5, "max": 20},
    "backchannel.max_gap": {"group": "Backchannel", "label": "Max gap (s)", "type": "float", "min": 0.5, "max": 30},
    "backchannel.hum_ratio": {"group": "Backchannel", "label": "Share of hums (vs words)", "type": "float",
                              "min": 0, "max": 1},
    "backchannel.gain": {"group": "Backchannel", "label": "Volume (0–1)", "type": "float", "min": 0, "max": 1},
    "backchannel.max_per_turn": {"group": "Backchannel", "label": "Max per user turn", "type": "int",
                                 "min": 0, "max": 10},
    "prompt": {"group": "Prompt", "label": "Main prompt", "type": "text"},
    # cohort_prompts.<COHORT> entries are added per agent (see _schema_for).
}

PREVIEW_TEXT = {
    "hin": "नमस्ते, मैं Gyan TV से Genie बोल रही हूँ। क्या आपके पास दो मिनट हैं? बस एक छोटा सा feedback लेना था।",
    "tel": "నమస్తే, నేను Gyan TV నుంచి Genie మాట్లాడుతున్నాను. మీకు రెండు నిమిషాలు ఉన్నాయా?",
    "tam": "வணக்கம், நான் Gyan TV-யிலிருந்து Genie பேசுகிறேன். உங்களுக்கு இரண்டு நிமிடங்கள் இருக்கிறதா?",
    "kan": "ನಮಸ್ಕಾರ, ನಾನು Gyan TV ಇಂದ Genie ಮಾತಾಡ್ತಾ ಇದ್ದೀನಿ. ನಿಮಗೆ ಎರಡು ನಿಮಿಷ ಇದೆಯಾ?",
    "mal": "നമസ്കാരം, ഞാൻ Gyan TV-യിൽ നിന്ന് Genie ആണ്. രണ്ട് മിനിറ്റ് സമയമുണ്ടോ?",
    "ben": "নমস্কার, আমি Gyan TV থেকে Genie বলছি। আপনার কি দুই মিনিট সময় হবে?",
    "mar": "नमस्कार, मी Gyan TV मधून Genie बोलतेय. तुमच्याकडे दोन मिनिटं आहेत का?",
    "eng": "Hi, this is Genie from Gyan TV. Do you have two minutes for some quick feedback?",
}


# ----- auth -----

def _password() -> str:
    return os.getenv("ADMIN_PASSWORD", "")


def _token() -> str:
    return hmac.new(_password().encode(), b"genie-admin-v1", hashlib.sha256).hexdigest()


def require_admin(request: Request) -> None:
    if not _password():
        raise HTTPException(503, "admin disabled — set ADMIN_PASSWORD")
    if not hmac.compare_digest(request.cookies.get(COOKIE, ""), _token()):
        raise HTTPException(401, "login required")


class LoginBody(BaseModel):
    password: str


@router.post("/login")
async def login(body: LoginBody, response: Response):
    if not _password():
        raise HTTPException(503, "admin disabled — set ADMIN_PASSWORD")
    if not hmac.compare_digest(body.password, _password()):
        await asyncio.sleep(1)   # slow down guessing
        raise HTTPException(401, "wrong password")
    response.set_cookie(COOKIE, _token(), httponly=True, samesite="strict", secure=True,
                        max_age=14 * 86400)
    return {"ok": True}


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie(COOKIE)
    return {"ok": True}


# ----- settings -----

def _get(d: dict, path: str):
    for part in path.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def _set(d: dict, path: str, value) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def _schema_for(agent: dict) -> dict[str, dict]:
    schema = dict(SCHEMA)
    for cohort in agent.get("cohorts") or []:
        schema[f"cohort_prompts.{cohort}"] = {"group": "Prompt", "label": f"Extra prompt for {cohort}",
                                              "type": "text"}
    for theme in agent.get("theme_prompts") or {}:
        schema[f"theme_prompts.{theme}"] = {"group": "Prompt", "label": f"Extra probes when the video's theme is {theme}",
                                            "type": "text"}
    return schema


def _coerce(path: str, spec: dict, value):
    t = spec["type"]
    try:
        if t == "bool":
            return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "on", "yes")
        if t in ("int", "float"):
            v = int(value) if t == "int" else float(value)
            if not spec.get("min", v) <= v <= spec.get("max", v):
                raise ValueError(f"must be between {spec['min']} and {spec['max']}")
            return v
        if t == "enum":
            if value not in spec["options"]:
                raise ValueError(f"must be one of {', '.join(spec['options'])}")
            return value
        v = str(value)
        if t == "str" and len(v) > 300:
            raise ValueError("too long (max 300 chars)")
        if t == "text" and len(v) > 20000:
            raise ValueError("too long")
        return v
    except (TypeError, ValueError) as e:
        raise HTTPException(400, f"{spec['label']}: {e}") from None


@router.get("/agents")
async def list_agents(request: Request):
    require_admin(request)
    effective, yaml_only = agents_mod.load_agents(), agents_mod.load_agents(overrides=False)
    out = []
    for agent_id, agent in effective.items():
        schema = _schema_for(agent)
        out.append({
            "id": agent_id, "name": agent["name"], "file": agent["file"], "cohorts": agent["cohorts"],
            "schema": schema,
            "values": {k: _get(agent, k) for k in schema},
            "yaml": {k: _get(yaml_only.get(agent_id, {}), k) for k in schema},
            "overrides": store.get_agent_overrides(agent_id),
        })
    return out


class SaveBody(BaseModel):
    values: dict = Field(default_factory=dict)   # dotted path → value


@router.put("/agents/{agent_id}")
async def save_agent(agent_id: str, body: SaveBody, request: Request):
    require_admin(request)
    yaml_agent = agents_mod.load_agents(overrides=False).get(agent_id)
    if not yaml_agent:
        raise HTTPException(404, "agent not found")
    schema = _schema_for(yaml_agent)
    overrides = store.get_agent_overrides(agent_id)
    for path, value in body.values.items():
        if path not in schema:
            raise HTTPException(400, f"unknown setting {path!r}")
        value = _coerce(path, schema[path], value)
        if value == _get(yaml_agent, path):
            # Same as the YAML → drop the override instead of pinning it.
            parent = _get(overrides, path.rsplit(".", 1)[0]) if "." in path else overrides
            if isinstance(parent, dict):
                parent.pop(path.rsplit(".", 1)[-1], None)
        else:
            _set(overrides, path, value)
    overrides = {k: v for k, v in overrides.items() if v != {}}
    bc = agents_mod._merge(yaml_agent.get("backchannel") or {}, overrides.get("backchannel") or {})
    if float(bc.get("min_gap", 0)) > float(bc.get("max_gap", 99)):
        raise HTTPException(400, "Backchannel: min gap must be ≤ max gap")
    store.set_agent_overrides(agent_id, overrides)
    return {"ok": True, "overrides": overrides}


@router.delete("/agents/{agent_id}")
async def reset_agent(agent_id: str, request: Request):
    require_admin(request)
    store.set_agent_overrides(agent_id, {})
    return {"ok": True}


# ----- voice preview -----

class PreviewBody(BaseModel):
    provider: str = "chirp"
    voice: str = "Sulafat"
    voice_id: str = ""      # ElevenLabs
    model: str = ""
    style: str = ""
    speed: float = 1.0
    language: str = "hin"
    text: str = ""


def _chirp(text: str, lang: str, voice: str, speed: float) -> bytes:
    from google.cloud import texttospeech as tts

    code = agents_mod.LANG_CODES[lang]
    resp = tts.TextToSpeechClient().synthesize_speech(
        input=tts.SynthesisInput(text=text),
        voice=tts.VoiceSelectionParams(language_code=code, name=f"{code}-Chirp3-HD-{voice}"),
        audio_config=tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16, sample_rate_hertz=24000,
                                     speaking_rate=speed),
    )
    return resp.audio_content   # LINEAR16 comes as a complete WAV file


@router.post("/preview")
async def preview(body: PreviewBody, request: Request):
    require_admin(request)
    if body.provider != "eleven" and body.voice not in VOICES:
        raise HTTPException(400, "unknown voice")
    lang = body.language if body.language in agents_mod.LANG_NAMES else "hin"
    text = (body.text or PREVIEW_TEXT[lang]).strip()[:400]
    try:
        if body.provider == "chirp":
            wav = await asyncio.to_thread(_chirp, text, lang, body.voice, max(0.5, min(1.5, body.speed)))
        elif body.provider == "eleven":
            from .eleven_tts import synthesize as eleven_synthesize, DEFAULT_MODEL
            voice_id = (body.voice_id or os.getenv("ELEVENLABS_VOICE_ID") or "").strip()
            if not voice_id:
                raise HTTPException(400, "ElevenLabs voice id missing")
            pcm, _ = await eleven_synthesize(text, voice_id=voice_id, model=(body.model or DEFAULT_MODEL).strip(),
                                             language=agents_mod.LANG_CODES[lang].split("-")[0])
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(pcm)
            wav = buf.getvalue()
        elif body.provider in GEMINI_TTS_MODELS:
            pcm = await gemini_synthesize(text, model=GEMINI_TTS_MODELS[body.provider],
                                          voice=body.voice, style=body.style[:300])
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(pcm)
            wav = buf.getvalue()
        else:
            raise HTTPException(400, "unknown provider")
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"preview failed: {e}") from None
    return Response(wav, media_type="audio/wav")
