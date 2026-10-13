"""Post-call analysis — turn a transcript into the agent's `analysis` JSON fields.

Runs once after each call with Gemini (google-genai, JSON mode). Cheap: one
request per call, a few hundred tokens.
"""

from __future__ import annotations

import json
import os

from google import genai
from google.genai import types
from loguru import logger

ANALYSIS_MODEL = os.getenv("ANALYSIS_MODEL", "gemini-3.1-flash-lite")


def transcript_text(turns: list[dict]) -> str:
    return "\n".join(f"{'User' if t['role'] == 'user' else 'Genie'}: {t['text']}" for t in turns)


async def analyse_call(agent: dict, turns: list[dict]) -> dict:
    fields: dict = agent.get("analysis") or {}
    if not fields or not turns:
        return {}
    schema_lines = "\n".join(f'- "{k}": {v}' for k, v in fields.items())
    prompt = (
        "You are analysing a phone-call transcript between Genie (an AI caller for Gyan TV) "
        "and a user. The user may speak Hindi/Telugu/Tamil/Kannada mixed with English.\n\n"
        f"Return ONLY a JSON object with exactly these keys:\n{schema_lines}\n\n"
        "Use booleans for true/false fields, arrays for list fields, and short English "
        "strings otherwise. If something was not discussed, use \"unclear\", false or [] "
        "as appropriate — never invent.\n\nTRANSCRIPT:\n" + transcript_text(turns)
    )
    try:
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        resp = await client.aio.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1),
        )
        data = json.loads(resp.text or "{}")
        return data if isinstance(data, dict) else {"raw": data}
    except Exception as e:
        logger.warning(f"[analysis] failed: {e}")
        return {"error": str(e)}
