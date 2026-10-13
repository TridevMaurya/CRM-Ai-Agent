"""Agent definitions — one YAML file per agent in agents/.

An agent = prompt template + voice/language defaults + the fields the post-call
analysis should extract. Per-call context (user name, video title, …) is filled
into `{{placeholders}}` at call time.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from . import store

from .paths import AGENTS_DIR  # agents/*.yaml (calling-agent/agents in the repo layout)

LANG_NAMES = {
    "hin": "Hindi", "tel": "Telugu", "tam": "Tamil",
    "kan": "Kannada", "mal": "Malayalam", "ben": "Bengali", "mar": "Marathi", "eng": "English",
}
# Picker code → Google locale (STT/TTS voice ids).
LANG_CODES = {
    "hin": "hi-IN", "tel": "te-IN", "tam": "ta-IN",
    "kan": "kn-IN", "mal": "ml-IN", "ben": "bn-IN", "mar": "mr-IN", "eng": "en-IN",
}

_PLACEHOLDER = re.compile(r"{{\s*([a-zA-Z0-9_]+)\s*}}")


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_agents(overrides: bool = True) -> dict[str, dict]:
    """Read every agents/*.yaml (re-read on each call so edits need no restart), with
    the admin panel's saved overrides layered on top (`overrides=False`: YAML only)."""
    agents: dict[str, dict] = {}
    for path in sorted(AGENTS_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        agent_id = data.get("id") or path.stem
        if overrides:
            data = _merge(data, store.get_agent_overrides(agent_id))
        data["id"] = agent_id
        data.setdefault("name", agent_id)
        data.setdefault("language", "hin")
        data.setdefault("voice", "Aoede")
        data.setdefault("speed", 1.0)
        data.setdefault("temperature", 0.7)
        data.setdefault("max_call_secs", 240)
        data.setdefault("greeting_first", True)
        data.setdefault("analysis", {})
        data.setdefault("defaults", {})
        data.setdefault("exotel", {}) or {}
        data["tts"] = {"provider": "chirp", "style": "", **(data.get("tts") or {})}
        data["cohorts"] = [str(c).upper() for c in (data.get("cohorts") or [])]
        data.setdefault("cohort_prompts", {})
        data["fillers"] = {"enabled": False, "every": 2, "hums_only": False, "gain": 0.8, "delay_secs": 0.0,
                           **(data.get("fillers") or {})}
        bc = {"enabled": True, "min_gap": 4.0, "max_gap": 6.0, "hum_ratio": 0.7,
              "max_per_turn": 3, "gain": 0.5}
        if "backchannel" in data["fillers"] and "backchannel" not in data:
            bc["enabled"] = bool(data["fillers"]["backchannel"])   # legacy alias (one release)
        data["backchannel"] = {**bc, **(data.get("backchannel") or {})}
        data["interruptions"] = {"enabled": False, "min_words": 2, **(data.get("interruptions") or {})}
        data.setdefault("silence_hangup_secs", 60)
        data["file"] = path.name
        agents[agent_id] = data
    return agents


def get_agent(agent_id: str) -> dict | None:
    return load_agents().get(agent_id)


def agent_for_cohort(cohort: str) -> dict | None:
    """The agent whose YAML lists this cohort under `cohorts:`."""
    cohort = (cohort or "").strip().upper()
    return next((a for a in load_agents().values() if cohort in a["cohorts"]), None)


_LANG_ALIASES = {
    "hindi": "hin", "hi": "hin", "telugu": "tel", "te": "tel", "tamil": "tam", "ta": "tam",
    "kannada": "kan", "kn": "kan", "malayalam": "mal", "ml": "mal", "bengali": "ben",
    "bangla": "ben", "bn": "ben", "marathi": "mar", "mr": "mar", "english": "eng", "en": "eng",
}


def normalize_language(value: str) -> str:
    """'Hindi' / 'hi' / 'hi-IN' / 'hin' → 'hin'. Unknown → '' (agent default applies)."""
    v = (value or "").strip().lower().split("-")[0]
    return v if v in LANG_NAMES else _LANG_ALIASES.get(v, "")


def resolve_language(agent: dict, params: dict) -> str:
    lang = (params.get("language") or params.get("lang") or agent["language"] or "hin").strip().lower()
    return lang if lang in LANG_NAMES else "hin"


# Appended to every agent prompt. Without it Gemini tends to write Telugu/Tamil in
# Roman letters ("nenu Gyan TV nunchi…"), which the TTS then reads as English.
_SCRIPT_RULE = (
    "\n\n[LANGUAGE] Speak in {name}. Write every {name} word in {script} — never "
    "romanised / Latin letters. English words that people naturally mix in (video, app, "
    "feedback, thank you) may stay in English letters. Numbers as digits."
)
# Appended when the user called us (inbound) instead of Genie calling them.
_INBOUND_NOTE = (
    "\n\n[CALL DIRECTION] This is an INCOMING call: the user called the Gyan TV number back "
    "after a missed call from Genie. They often don't know who this is (\"kaun?\", \"kaun sa?\", "
    "\"kyun call kiya tha?\") — answer that in one line (aapko Gyan TV se call aaya tha, main "
    "Genie, bas aapke dekhe video ke baare mein aapki raay chahiye) and continue with the same "
    "conversation. Do not say you are calling them."
)
# With fillers on, a short "hmm / achha" is played the moment the user stops
# talking; the reply must not repeat it.
_FILLER_NOTE = (
    "\n\n[ACKNOWLEDGEMENT] The phone system sometimes plays a short \"hmm / achha / "
    "okay / haan ji / theek hai / samajh gayi\" for you as soon as the user stops talking, "
    "and soft \"hmm\" sounds while they talk. Do not start your reply with any of those "
    "— go straight to the point. If the user's transcript contains a stray lone "
    "\"hmm\" or \"achha\", ignore it (it is your own sound echoing back)."
)
_SCRIPTS = {
    "hin": "Devanagari script (हिंदी)", "tel": "Telugu script (తెలుగు)",
    "tam": "Tamil script (தமிழ்)", "kan": "Kannada script (ಕನ್ನಡ)",
    "mal": "Malayalam script (മലയാളം)", "ben": "Bengali script (বাংলা)",
    "mar": "Devanagari script (मराठी)",
    "eng": "English",
}


def render_prompt(agent: dict, params: dict) -> str:
    """Fill {{placeholders}} from per-call params and append the script rule.

    Convenience placeholders: {{language_name}}, and for any key `x`,
    `{{x_or_<fallback words>}}` resolves to params[x] if present, else the
    fallback words with underscores turned into spaces (e.g.
    {{user_name_or_a_viewer}} → "a viewer").
    """
    lang = resolve_language(agent, params)
    values = {k: str(v) for k, v in params.items() if v is not None}
    values["language"] = lang
    values["language_name"] = LANG_NAMES[lang]
    if values.get("video_title"):
        # The model repeats the title aloud; give it the speakable form ("Ep 1:" and
        # thumbnail dashes dropped — see pipeline.speech_title).
        from .pipeline import speech_title
        values["video_title"] = speech_title(values["video_title"])

    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key in values and values[key].strip():
            return values[key]
        if "_or_" in key:
            base, fallback = key.split("_or_", 1)
            if base in values and values[base].strip():
                return values[base]
            return fallback.replace("_", " ")
        return ""

    # Fixed one-line answers to what users ask (YAML `user_questions`), rendered as a
    # list where the prompt says {{user_questions}}.
    uq = agent.get("user_questions") or []
    values["user_questions"] = "\n".join(
        f"  - {str(q.get('ask', '')).strip()} → {str(q.get('say', '')).strip()}" for q in uq if isinstance(q, dict)
    ) or "  (none)"
    prompt = _PLACEHOLDER.sub(sub, agent["prompt"]).strip()
    cohort = str(params.get("cohort", "")).strip().upper()
    block = (agent.get("cohort_prompts") or {}).get(cohort)
    if block:
        prompt += f"\n\n[THIS USER: {cohort}]\n" + _PLACEHOLDER.sub(sub, block).strip()
    # Theme-specific probes (from the research team's discussion guides): keyed by the
    # primary theme of the video they watched (params.theme, see themes.py).
    theme = str(params.get("theme", "")).strip()
    tblock = (agent.get("theme_prompts") or {}).get(theme)
    if tblock:
        prompt += f"\n\n[THEME: {theme}]\n" + _PLACEHOLDER.sub(sub, tblock).strip()
    if agent["fillers"]["enabled"] or agent["backchannel"]["enabled"]:
        prompt += _FILLER_NOTE
    prompt += _SCRIPT_RULE.format(name=LANG_NAMES[lang], script=_SCRIPTS[lang])
    if str(params.get("direction", "")).lower() == "inbound":
        prompt += _INBOUND_NOTE
    return prompt
