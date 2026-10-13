"""Automated end-to-end call against a running server, using the browser endpoint.

Speaks as the user with Sarvam-rendered audio (so no mic is needed), lets Genie
run the full feedback conversation, and prints the transcript, cost and latency.
Useful as a smoke test after every change and before deploying.

Usage: venv/bin/python scripts/e2e_browser_call.py [base_url] [agent_id] [lang] [k=v ...]
  extra k=v pairs become per-call params, e.g. tts=gemini_lite backchannel=0
  test-only keys: barge=1 (answer turn 3 while Genie is still talking — barge-in),
  double=1 (two user turns 2 s apart before Genie answers — expect ONE reply),
  wait_end=<secs> (how long to stay on after the last line; for the silence hang-up)
  lang hin_silent = say nothing after the greeting (use with silence_hangup_secs=10)

Also reports gaps inside Genie's audio (0.3–1.6 s holes between audio chunks) —
with Gemini TTS a reply must play without holes (a filler → reply gap is expected).
"""

import asyncio
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

import websockets
from dotenv import dotenv_values
from pipecat.frames.protobufs import frames_pb2

ROOT = Path(__file__).resolve().parent.parent   # call-server/
_ENV = next((d / ".env" for d in (ROOT, ROOT.parent, ROOT.parent.parent) if (d / ".env").is_file()), ROOT / ".env")
ENV = dotenv_values(_ENV)
BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080").rstrip("/")
AGENT = sys.argv[2] if len(sys.argv) > 2 else "genie-content-feedback"
LANG = sys.argv[3] if len(sys.argv) > 3 else "hin"
EXTRA = dict(a.split("=", 1) for a in sys.argv[4:] if "=" in a)
BARGE = EXTRA.pop("barge", "0") == "1"
# double=1: say turn 2, pause 2 s (turn ends, Genie starts thinking), then say turn 3
# before she answers — she must give ONE reply covering both, not two back to back.
DOUBLE = EXTRA.pop("double", "0") == "1"
WAIT_END = float(EXTRA.pop("wait_end", "8"))
RATE = 16000
CHUNK = RATE // 50 * 2  # 20 ms

# What the "user" says, turn by turn (Genie asks, we answer).
USER_TURNS = {
    "hin": ("hi-IN", [
        "हाँ बोलिए, एक मिनट है।",
        "हाँ हाँ, वो Bangalore वाला horror video देखा था।",
        "video ठीक था, लेकिन थोड़ा लंबा लगा और आखिर में कुछ खास नहीं हुआ।",
        "मुझे लगा था कोई real story होगी, पर वो बस बनाई हुई कहानी थी।",
        "ऐसी real incidents पर और video बनाइए, छोटे वाले।",
        "ठीक है, thank you।",
    ]),
    # Backchannel check: one ~20 s answer spoken without stopping.
    "hin_long": ("hi-IN", [
        "हाँ बोलिए, एक मिनट है।",
        "देखिए, वो Bangalore वाला video मैंने पूरा देखा था, शुरू में तो कहानी काफी अच्छी लगी, "
        "जिस तरह से उन्होंने पुराने घर का माहौल दिखाया, आवाज़ें और background music भी ठीक था, "
        "लेकिन बीच में जाकर video थोड़ा धीमा हो गया, बार बार वही बातें दोहराई गईं, "
        "और आखिर में जो twist था वो मुझे पहले से ही समझ आ गया था, तो मज़ा थोड़ा कम हो गया, "
        "अगर इसे पाँच छह मिनट छोटा कर देते तो और अच्छा लगता।",
        "ठीक है, thank you।",
    ]),
    "hin_silent": ("hi-IN", []),
    "tel": ("te-IN", [
        "అవును చెప్పండి, ఒక నిమిషం ఉంది.",
        "అవును, ఆ Bangalore horror video చూశాను.",
        "video బాగుంది కానీ కొంచెం పొడవుగా అనిపించింది.",
        "నిజమైన కథ అనుకున్నాను, కానీ అది కల్పితం.",
        "నిజ సంఘటనల మీద చిన్న videos చేయండి.",
        "సరే, thank you.",
    ]),
}


def tts(lang_code: str, text: str) -> bytes:
    req = urllib.request.Request(
        "https://api.sarvam.ai/text-to-speech", method="POST",
        headers={"api-subscription-key": ENV["SARVAM_API_KEY"], "Content-Type": "application/json"},
        data=json.dumps({"text": text, "target_language_code": lang_code, "speaker": "shubh",
                         "model": "bulbul:v3", "speech_sample_rate": RATE}).encode())
    wav = base64.b64decode(json.load(urllib.request.urlopen(req, timeout=60))["audios"][0])
    return wav[44:]


def audio_msg(pcm: bytes) -> bytes:
    return frames_pb2.Frame(audio=frames_pb2.AudioRawFrame(
        audio=pcm, sample_rate=RATE, num_channels=1)).SerializeToString()


async def main():
    code, lines = USER_TURNS[LANG]
    clips = [tts(code, t) for t in lines]
    print(f"[test] {len(clips)} user clips rendered")
    ws_url = BASE.replace("https://", "wss://").replace("http://", "ws://") + f"/ws/browser/{AGENT}"
    state = {"last_audio": 0.0, "bot_turns": 0, "closed": False, "reply_at": None,
             "audio_until": 0.0}
    lat, gaps = [], []

    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(json.dumps({"params": {"language": LANG.split("_")[0], "user_name": "Rahul",
                                             "video_title": "Bangalore horror story", **EXTRA}}))
        print(f"[test] connected {ws_url}")
        state["connected_at"] = time.monotonic()

        async def reader():
            try:
                async for msg in ws:
                    now = time.monotonic()
                    if isinstance(msg, str):
                        m = json.loads(msg)
                        print(f"  {'GENIE' if m['type'] == 'bot' else 'USER '}: {m['text']}")
                        if m["type"] == "bot":
                            state["bot_turns"] += 1
                            if "double_bot_turns" in state and "double_done" not in state:
                                state["double_replies"] = state["bot_turns"] - state["double_bot_turns"]
                        continue
                    f = frames_pb2.Frame()
                    f.ParseFromString(msg)
                    if f.HasField("audio") and f.audio.audio:
                        if "first_audio" not in state:
                            state["first_audio"] = now - state["connected_at"]
                            print(f"[test] first Genie audio {state['first_audio']:.2f}s after connect")
                        if state["reply_at"] is not None:
                            lat.append(now - state["reply_at"]); state["reply_at"] = None
                        # Audio arrives ~real time; a hole = this chunk starts well after
                        # the previous one finished playing.
                        hole = now - state["audio_until"]
                        if 0.3 < hole < 1.6 and state["reply_at"] is None:
                            gaps.append(hole)
                        secs = len(f.audio.audio) / 2 / (f.audio.sample_rate or 24000)
                        state["audio_until"] = max(now, state["audio_until"]) + secs
                        if state.get("talk_start") == float("inf"):
                            state["talk_start"] = now
                        state["last_audio"] = now
            except websockets.ConnectionClosed:
                pass
            state["closed"] = True

        reader_task = asyncio.create_task(reader())

        async def until_genie_quiet(max_wait=45, quiet=1.6):
            start = time.monotonic()
            while time.monotonic() - start < max_wait and not state["closed"]:
                try:
                    await ws.send(audio_msg(b"\x00" * CHUNK))
                except websockets.ConnectionClosed:
                    return
                await asyncio.sleep(0.02)
                if state["last_audio"] and time.monotonic() - state["last_audio"] > quiet:
                    return
            return

        async def until_genie_talking(after=1.2, max_wait=20):
            start = time.monotonic()
            while time.monotonic() - start < max_wait and not state["closed"]:
                await ws.send(audio_msg(b"\x00" * CHUNK))
                await asyncio.sleep(0.02)
                if state["last_audio"] and time.monotonic() - state["talk_start"] > after:
                    return

        await until_genie_quiet()
        for n, clip in enumerate(clips):
            if state["closed"]:
                break
            if BARGE and n == 3:
                # Talk over Genie's reply to turn 2 instead of waiting for her to finish.
                state["barge_sent_at"] = time.monotonic()
                print(f"[test] barging in over Genie")
            try:
                for i in range(0, len(clip), CHUNK):
                    await ws.send(audio_msg(clip[i:i + CHUNK]))
                    await asyncio.sleep(0.02)
            except websockets.ConnectionClosed:
                break
            state["reply_at"] = time.monotonic()
            state["last_audio"] = 0.0
            if DOUBLE and n == 1:
                # 2 s of silence: turn 2 ends and Genie starts on a reply; turn 3 follows
                # straight away (the loop's next iteration sends it without waiting).
                state["double_bot_turns"] = state["bot_turns"]
                print("[test] double: second utterance follows in 2 s")
                for _ in range(100):
                    await ws.send(audio_msg(b"\x00" * CHUNK))
                    await asyncio.sleep(0.02)
                continue
            if BARGE and n == 2:
                state["talk_start"] = float("inf")
                while state["talk_start"] == float("inf") and not state["closed"]:
                    await ws.send(audio_msg(b"\x00" * CHUNK))
                    await asyncio.sleep(0.02)
                await until_genie_talking()
                continue
            await until_genie_quiet()
            if DOUBLE and n == 2:
                state["double_done"] = True
        # Give Genie's hang-up a moment.
        for _ in range(int(WAIT_END / 0.02)):
            if state["closed"]:
                break
            try:
                await ws.send(audio_msg(b"\x00" * CHUNK))
            except websockets.ConnectionClosed:
                break
            await asyncio.sleep(0.02)
        reader_task.cancel()

    print(f"[test] closed by server: {state['closed']}  genie turns: {state['bot_turns']}")
    if DOUBLE:
        print(f"[test] double: Genie replies after the two quick turns (before the next user turn): "
              f"{state.get('double_replies', '?')}")
    if lat:
        print(f"[test] client-side voice-to-voice: " + ", ".join(f"{x:.2f}s" for x in lat))
    print(f"[test] holes in Genie audio (0.3–1.6 s): "
          + (", ".join(f"{x:.2f}s" for x in gaps) or "none"))
    await asyncio.sleep(6)  # analysis runs after the call
    calls = json.load(urllib.request.urlopen(f"{BASE}/api/calls?limit=1"))
    c = calls[0]
    print(f"[test] call {c['id']}: {c['status']} · {c['duration_secs']:.0f}s · ₹{c['cost_inr']:.2f} "
          f"(₹{(c['cost_detail'] or {}).get('inr_per_min')}/min) · p50 {c['latency_p50']}")
    print("[test] analysis:", json.dumps(c["analysis"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(asyncio.wait_for(main(), timeout=300))
