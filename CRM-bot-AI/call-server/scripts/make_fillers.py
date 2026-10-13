"""Generate the filler clips ONCE and save them to fillers/<lang>/*.wav.

Calls never hit a TTS for fillers — app/fillers.py just plays these files.

    # ElevenLabs v3 (most natural "hmm"s; needs ELEVENLABS_API_KEY + a voice id)
    python scripts/make_fillers.py --provider elevenlabs --voice <voice_id>

    # Google Chirp3-HD, same voice as the agents (no voice mismatch)
    python scripts/make_fillers.py --provider chirp --voice Sulafat

    # only some languages, replace existing clips
    python scripts/make_fillers.py --provider chirp --langs hin,tel --force

Listen to the clips, delete the ones that sound off, re-run (existing files are
skipped unless --force). Output: 24 kHz 16-bit mono WAV, silence trimmed.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.request
import wave
from array import array
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent   # call-server/
sys.path.insert(0, str(ROOT))
_ENV = next((d / ".env" for d in (ROOT, ROOT.parent, ROOT.parent.parent) if (d / ".env").is_file()), ROOT / ".env")
load_dotenv(_ENV)

from app.agents import LANG_CODES  # noqa: E402

OUT_DIR = ROOT / "fillers"
RATE = 24000
MAX_SECS = 1.2   # a filler longer than this starts to feel like Genie is stalling

# Short acknowledgements only — they're played before Genie's real reply, so nothing
# that commits to an answer. file stem → text. ElevenLabs v3 reads the [tags].
PHRASES: dict[str, dict[str, str]] = {
    # Hindi, Indian phone-call style; Genie is female ("samajh gayi"). Only neutral
    # acknowledgements — nothing that agrees ("bilkul", "sahi") since the user may be
    # complaining.
    "hin": {"hmm": "हम्म्म...", "achha": "अच्छा...", "achha_2": "अच्छा!",
            "okay": "ओके...", "hmm_okay": "हम्म, ओके...", "ok_ji": "ओके जी...",
            "haan_ji": "हाँ जी...", "hmm_haan": "हम्म, हाँ...",
            "achha_achha": "अच्छा अच्छा...", "achha_ji": "अच्छा जी...", "oh_achha": "ओह, अच्छा...",
            "theek_hai": "ठीक है...", "achha_theek_hai": "अच्छा, ठीक है...",
            "samajh_gayi": "समझ गई...", "right": "राइट..."},
    "tel": {"hmm": "హ్మ్...", "avuna": "అవునా...", "sare": "సరే...", "okay": "ఓకే...",
            "alaga": "అలాగా..."},
    "tam": {"hmm": "ம்ம்...", "sari": "சரி...", "appadiya": "அப்படியா...", "okay": "ஓகே..."},
    "kan": {"hmm": "ಹ್ಮ್...", "sari": "ಸರಿ...", "howda": "ಹೌದಾ...", "okay": "ಓಕೆ..."},
    "mal": {"hmm": "ഉം...", "shari": "ശരി...", "aano": "ആണോ...", "okay": "ഓക്കെ..."},
    "ben": {"hmm": "হুম...", "achha": "আচ্ছা...", "thik_ache": "ঠিক আছে...", "ha": "হ্যাঁ..."},
    "mar": {"hmm": "हम्म्म...", "achha": "अच्छा...", "bara": "बरं...", "ho_ka": "हो का...",
            "okay": "ओके..."},
    "eng": {"hmm": "Hmm...", "i_see": "I see...", "okay": "Okay...", "right": "Right..."},
}
V3_TAG = "[thoughtful] "   # ElevenLabs v3 audio tag — a soft, listening tone


def eleven_tts(text: str, voice_id: str, model: str) -> bytes:
    key = os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVENLABS_TTS_API_KEY")
    if not key:
        sys.exit("set ELEVENLABS_API_KEY in .env")
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=pcm_{RATE}",
        data=json.dumps({"text": V3_TAG + text if model == "eleven_v3" else text,
                         "model_id": model}).encode(),
        headers={"xi-api-key": key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()          # raw 16-bit mono PCM at RATE


def chirp_tts(text: str, lang: str, voice: str, speed: float) -> bytes:
    from google.cloud import texttospeech as tts

    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(_ENV.parent / "gcp-service-account.json"))
    code = LANG_CODES[lang]
    resp = tts.TextToSpeechClient().synthesize_speech(
        input=tts.SynthesisInput(text=text),
        voice=tts.VoiceSelectionParams(language_code=code, name=f"{code}-Chirp3-HD-{voice}"),
        audio_config=tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16,
                                     sample_rate_hertz=RATE, speaking_rate=speed),
    )
    with wave.open(io.BytesIO(resp.audio_content)) as w:   # LINEAR16 comes with a WAV header
        return w.readframes(w.getnframes())


def speech_ms(pcm: bytes, level: int = 1500) -> int:
    """How much of the clip is actually voiced (20 ms windows above `level`)."""
    a, win = array("h", pcm), RATE // 50
    return 20 * sum(1 for i in range(0, len(a), win) if max(map(abs, a[i:i + win]), default=0) > level)


def trim_silence(pcm: bytes, threshold: int = 250, pad_ms: int = 40) -> bytes:
    samples = array("h", pcm)
    loud = [i for i, s in enumerate(samples) if abs(s) > threshold]
    if not loud:
        return pcm
    pad = RATE * pad_ms // 1000
    return samples[max(0, loud[0] - pad): loud[-1] + pad].tobytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["elevenlabs", "chirp"], default="chirp")
    ap.add_argument("--voice", help="ElevenLabs voice id, or Chirp3-HD voice name (default Sulafat)")
    ap.add_argument("--model", default="eleven_v3", help="ElevenLabs model (default eleven_v3)")
    ap.add_argument("--speed", type=float, default=0.9, help="Chirp speaking_rate")
    ap.add_argument("--langs", default=",".join(PHRASES), help="comma-separated, e.g. hin,tel")
    ap.add_argument("--force", action="store_true", help="overwrite existing clips")
    args = ap.parse_args()
    voice = args.voice or (os.getenv("ELEVENLABS_VOICE_ID") if args.provider == "elevenlabs" else "Sulafat")
    if not voice:
        sys.exit("--voice <ElevenLabs voice id> (or ELEVENLABS_VOICE_ID in .env)")

    for lang in [s.strip() for s in args.langs.split(",") if s.strip()]:
        out = OUT_DIR / lang
        out.mkdir(parents=True, exist_ok=True)
        for stem, text in PHRASES[lang].items():
            path = out / f"{stem}.wav"
            if path.exists() and not args.force:
                print(f"skip  {path.relative_to(ROOT)}")
                continue
            # Very short inputs sometimes come back as ~0.2 s of near-silence (Chirp
            # especially) — retry until the clip has real speech in it.
            for _ in range(8):
                pcm = (eleven_tts(text, voice, args.model) if args.provider == "elevenlabs"
                       else chirp_tts(text, lang, voice, args.speed))
                if speech_ms(pcm) >= 150 and len(trim_silence(pcm)) / 2 / RATE <= MAX_SECS:
                    break
            else:
                print(f"FAIL  {path.relative_to(ROOT)}  no usable audio for {text!r} — skipped")
                continue
            pcm = trim_silence(pcm)
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
                w.writeframes(pcm)
            print(f"wrote {path.relative_to(ROOT)}  {len(pcm) / 2 / RATE:.2f}s  {text}")


if __name__ == "__main__":
    main()
