"""QC a set of calls from their Exotel recordings: Gemini listens to each mp3 and
reports interruptions (Genie talking over an unfinished user), dead air, STT/LLM
misunderstandings and overall feel.

Usage: python scripts/qc_recordings.py "<calls-json-url or file>" [max]
       e.g. python scripts/qc_recordings.py "https://<app>/api/calls?theme=UPSC&range=today" 10
Needs GOOGLE_API_KEY in the environment (or .env)."""
import json
import os
import sys
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
BASE = os.getenv("APP_URL", "https://gyan-tv-crm-glapp8.fly.dev")
OUT = Path(os.getenv("QC_DIR", "/tmp/qc"))
MODEL = os.getenv("QC_MODEL", "gemini-3.1-flash-lite")

PROMPT = """You are QC-ing a Hindi phone call between "Genie" (an AI caller, female voice, from Gyan TV)
and a user. Listen to the whole recording and answer in JSON:
{
 "interruptions": <int: times Genie started talking while the user was clearly mid-sentence / still going>,
 "interruption_examples": [<up to 3 short quotes of what the user was saying when cut off>],
 "user_waited_awkwardly": <int: times there was a 3s+ silence where the user was waiting for Genie>,
 "misunderstood": [<up to 3 cases where Genie's reply did not match what the user said>],
 "repeated_ack": <int: times Genie opened a reply with the same stock phrase like "samajh gayi"/"theek hai">,
 "user_engaged": <true if the user actually gave opinions, false if they were confused/silent/hung up quickly>,
 "verdict": "<one line: was this a good, natural call? what was the biggest problem?>"
}"""


def fetch_json(src: str):
    if src.startswith("http"):
        return json.load(urllib.request.urlopen(src))
    return json.loads(Path(src).read_text())


def main(src: str, limit: int):
    OUT.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    calls = [c for c in fetch_json(src) if c.get("status") == "ended" and (c.get("duration_secs") or 0) >= 20][:limit]
    results = []
    for c in calls:
        cid = c["id"]
        mp3 = OUT / f"{cid[:8]}.mp3"
        if not mp3.exists():
            try:
                urllib.request.urlretrieve(f"{BASE}/api/calls/{cid}/recording", mp3)
            except Exception as e:  # noqa: BLE001
                print(cid[:8], "no recording:", e)
                continue
        resp = client.models.generate_content(
            model=MODEL,
            contents=[types.Part.from_bytes(data=mp3.read_bytes(), mime_type="audio/mpeg"), PROMPT],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        try:
            r = json.loads(resp.text)
        except ValueError:
            r = {"verdict": resp.text[:200]}
        r["id"] = cid[:8]; r["secs"] = c.get("duration_secs"); r["bucket"] = c.get("bucket")
        results.append(r)
        print(f"{cid[:8]} {int(c.get('duration_secs') or 0):>4}s cut={r.get('interruptions')} "
              f"wait={r.get('user_waited_awkwardly')} ack={r.get('repeated_ack')} | {r.get('verdict')}")
    (OUT / "qc.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    n = len(results) or 1
    print(f"\n{len(results)} calls · interruptions/call {sum(r.get('interruptions') or 0 for r in results) / n:.1f} · "
          f"awkward waits/call {sum(r.get('user_waited_awkwardly') or 0 for r in results) / n:.1f} · "
          f"repeated acks/call {sum(r.get('repeated_ack') or 0 for r in results) / n:.1f} · "
          f"engaged {sum(1 for r in results if r.get('user_engaged'))}/{len(results)}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10)
