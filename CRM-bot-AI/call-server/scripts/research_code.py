"""Code every real conversation with one research codebook (Gemini, JSON mode) so
the whole run can be counted consistently — the per-call analysis stored at call
time changed schema several times. Resumable: results are appended to a JSONL file.

Usage: python scripts/research_code.py <calls.pkl from the xlsx export> <out.jsonl> [concurrency]"""
import asyncio
import json
import os
import pickle
import sys
from pathlib import Path

from google import genai
from google.genai import types

MODEL = os.getenv("RESEARCH_MODEL", "gemini-3.1-flash-lite")
REASONS = ["unaware_auto_debit", "wanted_refund", "expected_to_earn_money", "content_not_useful_or_practical",
           "content_not_found", "content_quality", "language_barrier", "app_hard_to_use", "price_too_high",
           "no_money_right_now", "no_time_or_not_using", "technical_issue", "uses_free_alternative",
           "only_wanted_to_try", "someone_else_subscribed", "did_not_cancel_or_still_using", "other", "not_stated"]
PROMPT = """You are coding a phone survey for Gyan TV (an Indian short-video learning app with a paid
subscription). "Genie" is the AI caller; "User" is the person who picked up. Read the transcript and
return ONE JSON object. Code only what the USER actually said — never infer from Genie's questions.
If the user barely spoke, most fields are "" / [] / "not_stated".

Call context: cohort={cohort} (TRIAL_CANCELLED / SUB_CANCELLED = they cancelled; DAU_10PLUS / LIGHT_3_4 =
active viewers asked for content feedback) · theme they watch={theme} · last video="{video}"

JSON fields:
- "respondent": "user" | "family_member" | "wrong_person" | "unclear"
- "engagement": "full" (answered 2+ questions with content) | "partial" (one real answer) | "refused_or_busy" | "confused" (did not understand who/why) | "none"
- "knows_gyantv": "yes" | "no" | "unclear"   (do they recognise the app at all?)
- "aware_of_payment": "yes" | "no_didnt_know_charged" | "unclear"
- "cancel_reasons": array, any of {reasons}
- "main_reason": exactly one of the same list
- "amount_mentioned": money amount the user mentioned being charged / paying, e.g. "₹299", else ""
- "refund_requested": true | false
- "watched_content": "yes" | "no" | "unclear"
- "liked": what they liked about the videos/app, short English, else ""
- "disliked": what they disliked / found lacking, short English, else ""
- "disliked_tags": array, any of ["not_practical_no_steps","too_basic_or_generic","misleading_promise","too_long","too_short","poor_audio","poor_visuals","hard_language","repetitive","not_sequenced","hard_to_find_content","paywall_or_ads"]
- "content_requests": array of short English phrases — topics / kinds of video they asked for
- "request_categories": array, any of ["practical_earning_methods","govt_exam_prep","english_speaking","share_market_finance","business_ideas","ai_tech_skills","devotional","entertainment","health","agriculture","regional_language","school_education","history_gk","other"]
- "would_return": "yes" | "no" | "maybe" | "not_asked"
- "price_willing": price they said they would pay, else ""
- "sentiment": "positive" | "neutral" | "negative" | "angry"
- "quote_hi": one short verbatim user line (their words, as in the transcript) that best captures their point, else ""
- "quote_en": that line in English
- "one_line": one English sentence: who this was and what they told us
- "insight_score": 0 (nothing usable) | 1 (vague) | 2 (one specific point) | 3 (rich, actionable)

Transcript:
{transcript}"""


def user_words(t: str) -> int:
    return sum(len(l[5:].split()) for l in (t or "").split("\n") if l.startswith("User:"))


async def main(src: str, out: str, conc: int):
    data = pickle.load(open(src, "rb"))
    todo = [d for d in data if d["status"] == "ended" and user_words(d["transcript"]) >= 5]
    outp = Path(out)
    done = set()
    if outp.exists():
        done = {json.loads(l)["key"] for l in outp.read_text().splitlines() if l.strip()}
    todo = [d for d in todo if f'{d["phone"]}|{d["started"]}' not in done]
    print(f"{len(done)} done, {len(todo)} to code with {MODEL}", flush=True)
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    sem = asyncio.Semaphore(conc)
    lock = asyncio.Lock()
    n = 0

    async def one(d):
        nonlocal n
        prompt = PROMPT.format(cohort=d["cohort"] or "unknown", theme=d["theme"] or "unknown",
                               video=d["video_title"] or "", reasons=json.dumps(REASONS),
                               transcript=(d["transcript"] or "")[:12000])
        async with sem:
            for attempt in range(4):
                try:
                    r = await client.aio.models.generate_content(
                        model=MODEL, contents=prompt,
                        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0))
                    res = json.loads(r.text)
                    if isinstance(res, list):
                        res = res[0]
                    break
                except Exception as e:  # noqa: BLE001
                    res = {"error": str(e)[:200]}
                    await asyncio.sleep(2 * (attempt + 1))
        res["key"] = f'{d["phone"]}|{d["started"]}'
        async with lock:
            with outp.open("a") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            n += 1
            if n % 100 == 0:
                print(f"{n}/{len(todo)}", flush=True)

    await asyncio.gather(*(one(d) for d in todo))
    print("done", n, flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 12))
