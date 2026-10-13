"""Filler sounds that make Genie sound like she's listening.

1. Turn-end filler (`fillers:` in the agent YAML) — a short "okay / achha / haan ji"
   played the instant the user stops talking, on every Nth turn, so the gap while the
   LLM + TTS work isn't dead air.
2. Backchannel v2 (`backchannel:`) — while the user talks, a soft listening sound:
   - timer: every min_gap..max_gap seconds of speech (re-drawn each time, never
     metronomic), and
   - micro-pause: at a ~0.2 s dip in loudness (shorter than the VAD's end-of-speech)
     once ≥ 0.6 × min_gap of speech has passed since the last one.
   Mostly recorded hums (fillers/hums/, language-neutral), sometimes a word
   (fillers/<lang>/), at reduced gain so it sits under the caller's voice.

Clips are made ONCE (scripts/make_fillers.py; hums are recordings) and read from
memory at call time: no TTS calls, no added latency.

The player is a second audio channel: it sits right before transport.output (after
the TTS, like sahi-prompt-tester's BackchannelProcessor), so its clips never queue
behind the LLM or TTS, and go out in 20 ms chunks that an interruption can cut.
Everything is a plain OutputAudioRawFrame (not TTS audio), so the output transport
plays it without Bot{Started,Stopped}SpeakingFrame — it can't trigger the
end-of-call hang-up, the latency observer, or turn-taking. The gate ignores what the
STT hears while a turn-end filler plays, and drops a bare "hmm / achha" transcript
right after a backchannel (echo through the caller's handset — phones have no AEC).
"""

from __future__ import annotations

import asyncio
import random
import re
import time
import wave
from array import array
from functools import lru_cache
from pathlib import Path

import numpy as np
from loguru import logger
from dataclasses import dataclass

from pipecat.frames.frames import (
    DataFrame,
    InputAudioRawFrame,
    LLMContextFrame,
    OutputAudioRawFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .paths import FILLERS_DIR  # fillers/<lang>/*.wav + fillers/hums (calling-agent/fillers in the repo layout)
HUMS_DIR = FILLERS_DIR / "hums"
RATE = 24000

# Word clips for the backchannel "word" pool (the rest of the language's clips stay
# turn-end fillers). Languages not listed fall back to their shortest three clips.
BACKCHANNEL_WORDS = {
    "hin": ["achha", "okay", "theek_hai"],
    "tel": ["sare", "okay", "avuna"],
    "tam": ["sari", "okay", "appadiya"],
    "kan": ["sari", "okay", "howda"],
    "mal": ["shari", "okay", "aano"],
    "mar": ["bara", "okay", "ho_ka"],
}
MICRO_PAUSE_SECS = 0.18   # loudness dip that counts as a breath / pause
POLL_SECS = 0.2           # timer resolution
SPEECH_GAP_RESET = 1.5    # silence longer than this starts a fresh stretch of speech
CHUNK_BYTES = RATE * 2 // 50   # clips go out in 20 ms frames

# What a filler sounds like once it has leaked back through the handset and been
# transcribed. Used only right after a backchannel played.
_FILLER_TOKENS = {
    "hmm", "hm", "hmmm", "mm", "mmm", "uh-huh", "ok", "okay", "achha", "acha",
    "हम्म", "हम्म्म", "हम", "हूं", "हूँ", "म्म", "अच्छा", "ओके", "ठीक", "है", "हाँ", "हां", "जी",
    "హ్మ్", "సరే", "ఓకే", "అవును", "అవునా", "ம்ம்", "சரி", "ஓகே", "ಹ್ಮ್", "ಸರಿ", "ಓಕೆ", "ಹೌದು",
    "ഉം", "ശരി", "ഓക്കെ", "ഓകെ", "হুম", "আচ্ছা", "ওকে", "ঠিক", "আছে",
}
_PUNCT = re.compile(r"[\s.,!?।…\-–—\"'“”‘’()]+")


def is_filler_echo(text: str) -> bool:
    tokens = [t for t in _PUNCT.split((text or "").lower()) if t]
    return bool(tokens) and len(tokens) <= 3 and all(t in _FILLER_TOKENS for t in tokens)


def _read_wav(path: Path) -> bytes | None:
    try:
        with wave.open(str(path), "rb") as w:
            if w.getsampwidth() != 2 or w.getnchannels() != 1 or w.getframerate() != RATE:
                logger.warning(f"[fillers] {path.name}: need 16-bit mono {RATE} Hz, skipped")
                return None
            return w.readframes(w.getnframes())
    except (wave.Error, OSError) as e:
        logger.warning(f"[fillers] {path}: {e}")
        return None


def _decode_other(path: Path) -> bytes | None:
    """mp3 / m4a / ogg via PyAV when it's installed (it isn't in the image — commit
    hums as 24 kHz WAV; this is for dropping in a recording locally)."""
    try:
        import av
    except ImportError:
        logger.warning(f"[fillers] {path.name}: PyAV not installed, convert to WAV")
        return None
    container = av.open(str(path))
    resampler = av.AudioResampler(format="s16", layout="mono", rate=RATE)
    pcm = bytearray()
    for frame in container.decode(container.streams.audio[0]):
        for rf in resampler.resample(frame):
            pcm += rf.to_ndarray().tobytes()
    return trim_silence(bytes(pcm))


def trim_silence(pcm: bytes, thresh: float = 0.02, pad_ms: int = 60) -> bytes:
    samples = array("h", pcm)
    level = int(thresh * 32767)
    loud = [i for i, s in enumerate(samples) if abs(s) > level]
    if not loud:
        return pcm
    pad = RATE * pad_ms // 1000
    return samples[max(0, loud[0] - pad): loud[-1] + pad].tobytes()


@lru_cache(maxsize=None)
def load_clips(lang: str) -> tuple[tuple[str, bytes], ...]:
    """(name, pcm16 mono 24 kHz) for every fillers/<lang>/*.wav."""
    out = []
    for path in sorted((FILLERS_DIR / lang).glob("*.wav")):
        pcm = _read_wav(path)
        if pcm:
            out.append((path.stem, pcm))
    return tuple(out)


@lru_cache(maxsize=None)
def load_hums() -> tuple[tuple[str, bytes], ...]:
    out = []
    for path in sorted(HUMS_DIR.glob("*")):
        suffix = path.suffix.lower()
        pcm = (_read_wav(path) if suffix == ".wav"
               else _decode_other(path) if suffix in (".mp3", ".m4a", ".ogg") else None)
        if pcm:
            out.append((f"hums/{path.stem}", pcm))
    return tuple(out)


def _scale(pcm: bytes, gain: float) -> bytes:
    if gain >= 0.999:
        return pcm
    return (np.frombuffer(pcm, dtype=np.int16) * gain).astype(np.int16).tobytes()


def _secs(pcm: bytes) -> float:
    return len(pcm) / 2 / RATE


@dataclass
class PreRenderedSpeechFrame(DataFrame):
    """Genie speech synthesised ahead of time (the opening). Queued through the
    pipeline like any frame — so it plays after StartFrame, in order — and turned
    into TTS audio by the FillerPlayer just before the output."""

    pcm: bytes = b""


class FillerPlayer(FrameProcessor):
    """The second audio channel, right before transport.output. Passes every frame
    through and injects clips when asked:

    - turn_end(): from TurnEndTrigger when a user turn is handed to the LLM; plays on
      every `every`-th turn, skipped when a backchannel just played.
    - backchannel(): from SpeechWatcher while the user talks; picks hum vs word,
      applies gain and the per-turn cap."""

    def __init__(self, session, lang: str, *, turn_fillers: bool = True, every: int = 2,
                 backchannel: dict | None = None, hums_only: bool = False, filler_gain: float = 0.8,
                 delay_secs: float = 0.0):
        super().__init__()
        self._session = session
        self._delay = max(0.0, float(delay_secs))   # hum this long after the user stops
        # hums_only: the two recorded hums are the whole vocabulary — no word clips.
        if turn_fillers and hums_only:
            self._clips = tuple((n, _scale(p, filler_gain)) for n, p in load_hums())
        else:
            self._clips = load_clips(lang) if turn_fillers else ()
        self._every = max(1, int(every))
        self._turn = 0
        self._last = ""
        self._bc = backchannel or {}
        gain = max(0.0, min(1.0, float(self._bc.get("gain", 0.5))))
        if self._bc.get("enabled"):
            self._hums = tuple((n, _scale(p, gain)) for n, p in load_hums())
            all_words = load_clips(lang)
            wanted = BACKCHANNEL_WORDS.get(lang)
            words = [c for c in all_words if c[0] in wanted] if wanted else []
            words = words or sorted(all_words, key=lambda c: len(c[1]))[:3]
            self._words = tuple((n, _scale(p, gain)) for n, p in words)
        else:
            self._hums = self._words = ()
        self._bc_last_at = 0.0
        self._bc_count = 0
        if turn_fillers and not self._clips:
            logger.info(f"[fillers] no clips for {lang!r} in {FILLERS_DIR} — turn-end fillers off")

    def _pick(self, clips) -> tuple[str, bytes]:
        choices = [c for c in clips if c[0] != self._last] or list(clips)
        clip = random.choice(choices)
        self._last = clip[0]
        return clip

    async def _play(self, clip, kind: str):
        name, pcm = clip
        logger.info(f"[{self._session.call_id}] {kind} {name} ({_secs(pcm):.2f}s)")
        self._session.filler_playing = True
        for i in range(0, len(pcm), CHUNK_BYTES):
            await self.push_frame(OutputAudioRawFrame(audio=pcm[i:i + CHUNK_BYTES],
                                                      sample_rate=RATE, num_channels=1))

    async def _play_speech(self, pcm: bytes):
        """Pre-rendered Genie speech as TTS audio, so it counts as Genie speaking."""
        await self.push_frame(TTSStartedFrame())
        for i in range(0, len(pcm), CHUNK_BYTES):
            await self.push_frame(TTSAudioRawFrame(pcm[i:i + CHUNK_BYTES], RATE, 1))
        await self.push_frame(TTSStoppedFrame())

    async def backchannel(self, trigger: str) -> bool:
        """Play one listening sound. Returns False when a guard blocked it."""
        if (not (self._hums or self._words) or self._session.ending or self._session.bot_speaking
                or self._bc_count >= int(self._bc.get("max_per_turn", 3))):
            return False
        use_hum = bool(self._hums) and (not self._words
                                        or random.random() < float(self._bc.get("hum_ratio", 0.7)))
        clip = self._pick(self._hums if use_hum else self._words)
        self._bc_last_at, self._bc_count = time.monotonic(), self._bc_count + 1
        # Not an echo gate for everything (the caller is mid-sentence and their words
        # must count) — only a bare filler-word transcript in this window is dropped.
        self._session.backchannel_until = time.monotonic() + _secs(clip[1]) + 0.6
        await self._play(clip, f"backchannel[{trigger}]")
        return True

    async def turn_end(self):
        """A user turn was just handed to the LLM. On every `every`-th turn, hum
        `delay_secs` after the user stopped talking — unless Genie's reply has already
        started by then (then no hum is needed)."""
        self._bc_count = 0
        self._turn += 1
        recent_bc = time.monotonic() - self._bc_last_at < 1.5
        if not (self._clips and self._turn % self._every == 0 and not recent_bc
                and not self._session.ending and not self._session.bot_speaking):
            return
        wait = self._delay - (time.monotonic() - (self._session.last_vad_at or time.monotonic()))
        if wait > 0:
            await asyncio.sleep(wait)
            if self._session.bot_speaking or self._session.ending or self._session.user_speaking:
                return
        clip = self._pick(self._clips)
        self._session.filler_until = time.monotonic() + _secs(clip[1]) + 0.4
        await self._play(clip, "filler")

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, PreRenderedSpeechFrame):
            logger.info(f"[{self._session.call_id}] pre-rendered opening ({_secs(frame.pcm):.1f}s)")
            await self._play_speech(frame.pcm)
            return
        await self.push_frame(frame, direction)


class TurnEndTrigger(FrameProcessor):
    """Sits right after the user aggregator: each LLMContextFrame that follows real
    user speech is one finished user turn → tell the player (which is downstream of
    the TTS, so the filler plays while the LLM is still thinking)."""

    def __init__(self, player: FillerPlayer, session):
        super().__init__()
        self._player = player
        self._session = session

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and self._session.user_spoke and not self._session.ending:
            self._session.user_spoke = False
            asyncio.create_task(self._player.turn_end())   # don't hold the context frame
        await self.push_frame(frame, direction)


class SpeechWatcher(FrameProcessor):
    """Sits right after the VAD processor and passes every frame through. Keeps the
    time the current stretch of user speech started (since the last backchannel),
    runs the jittered timer while the user speaks, and spots micro-pauses."""

    def __init__(self, player: FillerPlayer, session, cfg: dict):
        super().__init__()
        self._player = player
        self._session = session
        self._min = float(cfg.get("min_gap", 4.0))
        self._max = max(float(cfg.get("max_gap", 6.0)), self._min)
        self._speaking = False
        self._since = 0.0            # start of speech since the last backchannel
        self._stopped_at = 0.0
        self._target = self._draw()
        self._timer: asyncio.Task | None = None
        self._speech_level = 0.0     # running loudness of the user's speech
        self._quiet = 0.0            # seconds of continuous quiet
        self._fired = False          # already fired for the current dip

    def _draw(self) -> float:
        return random.uniform(self._min, self._max)

    async def _fire(self, trigger: str):
        if await self._player.backchannel(trigger):
            self._since, self._target = time.monotonic(), self._draw()

    async def _run_timer(self):
        try:
            while self._speaking:
                await asyncio.sleep(POLL_SECS)
                if self._session.bot_speaking:
                    self._since = time.monotonic()
                elif self._speaking and time.monotonic() - self._since >= self._target:
                    await self._fire("timer")
        except asyncio.CancelledError:
            pass

    def _stop_timer(self):
        if self._timer and not self._timer.done():
            self._timer.cancel()
        self._timer = None

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            now = time.monotonic()
            if not self._speaking:
                if now - self._stopped_at > SPEECH_GAP_RESET or not self._since:
                    self._since = now
                self._speaking, self._quiet = True, 0.0
                self._timer = asyncio.create_task(self._run_timer())
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._speaking, self._stopped_at = False, time.monotonic()
            self._stop_timer()
        elif isinstance(frame, InputAudioRawFrame) and self._speaking and frame.audio:
            await self._on_audio(frame)
        await self.push_frame(frame, direction)

    async def _on_audio(self, frame: InputAudioRawFrame):
        samples = np.frombuffer(frame.audio[: len(frame.audio) - len(frame.audio) % 2], dtype=np.int16)
        if not samples.size:
            return
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        secs = samples.size / frame.sample_rate / max(1, frame.num_channels)
        if rms > max(400.0, 0.5 * self._speech_level):
            self._speech_level = rms if not self._speech_level else 0.95 * self._speech_level + 0.05 * rms
            self._quiet, self._fired = 0.0, False
            return
        if rms < max(250.0, 0.2 * self._speech_level):
            self._quiet += secs
            if (self._quiet >= MICRO_PAUSE_SECS and not self._fired
                    and time.monotonic() - self._since >= 0.6 * self._min):
                self._fired = True
                await self._fire("pause")

    async def cleanup(self):
        await super().cleanup()
        self._stop_timer()
