"""ElevenLabs Eleven v4 Turbo over the Text-to-Dialogue WebSocket.

v3/v4 models are NOT served by the classic text-to-speech websocket that pipecat's
ElevenLabsTTSService speaks (``/v1/text-to-speech/{voice}/stream-input`` rejects
``eleven_v4*``), only by

    wss://api.elevenlabs.io/v1/text-to-dialogue/stream-input?model_id=eleven_v4_turbo&output_format=pcm_24000

    first message   {"voices": ["<voice_id>"], "voice_settings": {"stability": 0.5}}
    each sentence   {"inputs": [{"text": "...", "voice_id": "<voice_id>", "new_turn": true}], "flush": true}
    server          {"audio": "<base64 pcm>"} … {"is_final_audio_for_turn": true}
                    {"keep_alive": true} every few seconds or the server closes after 20 s idle
                    {"error": "...", "message": "...", "code": 1008}

One socket per call (a dialogue session). pipecat hands sentences to run_tts one at
a time, so each sentence is sent with ``flush`` (the server otherwise buffers ~40
chars / 8 words) and read back until the end-of-turn marker. On an interruption the
socket is dropped — the server has no per-context cancel — and reopened lazily.

Billed per character ($11 / 1M on the Gyan TV enterprise rate, 8 Oct 2026).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from typing import AsyncGenerator

import websockets
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, StartFrame, EndFrame, CancelFrame, TTSAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

WS_URL = "wss://api.elevenlabs.io/v1/text-to-dialogue/stream-input"
DEFAULT_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_v4_turbo")
KEEPALIVE_SECS = 8.0          # server closes after 20 s without a client message
FIRST_AUDIO_SECS = 3.0        # no first byte by then: the socket is stuck — reconnect and resend once
TURN_TIMEOUT_SECS = 8.0       # give up on a sentence that never finishes (was 12: that is a 12 s silence on the call)
TAIL_IDLE_SECS = 1.5          # no end-of-turn marker: treat 1.5 s of silence after audio as the end
SLOW_TTFB_SECS = 1.5          # log a line slower than this (12 Oct QC: one call's audio arrived 18 s late)


class ElevenTTS(TTSService):
    def __init__(self, *, session, api_key: str, voice_id: str, model: str = DEFAULT_MODEL,
                 language: str = "hi", stability: float = 0.5, sample_rate: int = 24000, **kwargs):
        super().__init__(sample_rate=sample_rate,
                         settings=TTSSettings(model=model, voice=voice_id, language=None), **kwargs)
        self._call = session            # CallSession: text clean-up
        self._api_key = api_key or ""
        self._voice_id = voice_id
        self._model = model
        self._language = language
        self._stability = stability
        self._ws: websockets.ClientConnection | None = None
        self._last_sent = 0.0
        self._keepalive_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()     # one sentence on the wire at a time

    def can_generate_metrics(self) -> bool:
        return True

    # ----- socket lifecycle -----

    def _url(self) -> str:
        url = f"{WS_URL}?model_id={self._model}&output_format=pcm_{self.sample_rate}"
        if self._language:
            url += f"&language_code={self._language}"
        return url

    async def _connect(self) -> websockets.ClientConnection:
        if self._ws is not None:
            return self._ws
        t0 = time.monotonic()
        ws = await websockets.connect(self._url(), additional_headers={"xi-api-key": self._api_key},
                                      open_timeout=10, max_size=None)
        await ws.send(json.dumps({"voices": [self._voice_id],
                                  "voice_settings": {"stability": self._stability}}))
        self._ws = ws
        self._last_sent = time.monotonic()
        if self._keepalive_task is None or self._keepalive_task.done():
            self._keepalive_task = asyncio.create_task(self._keepalive())
        logger.debug(f"{self}: ElevenLabs dialogue socket open in {time.monotonic() - t0:.2f}s ({self._model})")
        return ws

    async def _close(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass

    async def _keepalive(self) -> None:
        try:
            while True:
                await asyncio.sleep(2.0)
                ws = self._ws
                if ws is None:
                    continue
                if time.monotonic() - self._last_sent >= KEEPALIVE_SECS and not self._lock.locked():
                    try:
                        await ws.send(json.dumps({"keep_alive": True}))
                        self._last_sent = time.monotonic()
                    except Exception as e:  # noqa: BLE001
                        logger.debug(f"{self}: keep-alive failed, dropping socket: {e}")
                        await self._close()
        except asyncio.CancelledError:
            pass

    async def start(self, frame: StartFrame):
        await super().start(frame)
        try:
            await self._connect()        # handshake before the greeting, not during it
        except Exception as e:  # noqa: BLE001
            logger.warning(f"{self}: ElevenLabs connect at start failed (will retry on first line): {e}")

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._shutdown()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._shutdown()

    async def _shutdown(self) -> None:
        if self._keepalive_task:
            self._keepalive_task.cancel()
            self._keepalive_task = None
        await self._close()

    async def _handle_interruption(self, frame, direction: FrameDirection):
        await super()._handle_interruption(frame, direction)
        # No per-sentence cancel on this socket: drop it so the half-generated line
        # can't play after the user's new turn; the next line reconnects (~0.3 s).
        await self._close()

    # ----- synthesis -----

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        text = self._call.clean_tts_text(text)
        if not text:
            yield None
            return
        logger.debug(f"{self}: ElevenLabs TTS [{text}]")
        async with self._lock:
            try:
                await self.start_ttfb_metrics()
                ws = await self._connect()
                await ws.send(json.dumps({"inputs": [{"text": text, "voice_id": self._voice_id, "new_turn": True}],
                                          "flush": True}))
                self._last_sent = time.monotonic()
                await self.start_tts_usage_metrics(text)
                got_audio = False
                retried = False
                started = time.monotonic()
                while True:
                    waited = time.monotonic() - started
                    if not got_audio and not retried and waited >= FIRST_AUDIO_SECS:
                        # Nothing back in 3 s: a stuck socket (seen after interruptions).
                        # A fresh socket answers in ~0.3 s; waiting out the old one is a
                        # long silence on the call.
                        logger.warning(f"{self}: ElevenLabs: no audio in {waited:.1f}s — reconnecting for {text[:40]!r}")
                        await self._close()
                        ws = await self._connect()
                        await ws.send(json.dumps({"inputs": [{"text": text, "voice_id": self._voice_id, "new_turn": True}],
                                                  "flush": True}))
                        self._last_sent = time.monotonic()
                        retried = True
                        started = time.monotonic()
                        continue
                    wait = TAIL_IDLE_SECS if got_audio else min(FIRST_AUDIO_SECS if not retried else TURN_TIMEOUT_SECS,
                                                                 TURN_TIMEOUT_SECS) - waited
                    if wait <= 0:
                        logger.warning(f"{self}: ElevenLabs: no audio within {TURN_TIMEOUT_SECS:.0f}s for {text[:40]!r}")
                        await self._close()
                        break
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=wait)
                    except asyncio.TimeoutError:
                        if got_audio:
                            break                # tail went quiet without a marker: done
                        continue
                    msg = json.loads(raw)
                    if msg.get("error") or msg.get("message") and msg.get("code"):
                        logger.warning(f"ElevenLabs TTS error: {msg}")
                        await self._close()
                        yield ErrorFrame(error=f"ElevenLabs TTS: {msg.get('message') or msg.get('error')}")
                        return
                    audio = msg.get("audio")
                    if audio:
                        if not got_audio:
                            await self.stop_ttfb_metrics()
                            got_audio = True
                            ttfb = time.monotonic() - started
                            self._call.observer.tts_ttfbs.append(ttfb)
                            if ttfb >= SLOW_TTFB_SECS:
                                logger.warning(f"[{self._call.call_id}] ElevenLabs slow: first audio after {ttfb:.1f}s"
                                               f"{' (after reconnect)' if retried else ''} for {text[:40]!r}")
                        yield TTSAudioRawFrame(base64.b64decode(audio), self.sample_rate, 1, context_id=context_id)
                    if msg.get("is_final_audio_for_turn") or msg.get("is_final"):
                        if msg.get("is_final"):
                            await self._close()  # server ended the session
                        break
                if not got_audio:
                    yield ErrorFrame(error="ElevenLabs TTS: no audio for line")
            except websockets.ConnectionClosed as e:
                logger.warning(f"ElevenLabs socket closed mid-line: {e}")
                await self._close()
                yield ErrorFrame(error=f"ElevenLabs TTS: socket closed ({e.code})")
            except Exception as e:  # noqa: BLE001 — one failed line must not end the call
                logger.warning(f"ElevenLabs TTS failed: {e}")
                await self._close()
                yield ErrorFrame(error=f"ElevenLabs TTS failed: {e}")


async def synthesize(text: str, *, voice_id: str, model: str = DEFAULT_MODEL, language: str = "hi",
                     sample_rate: int = 24000, api_key: str | None = None) -> tuple[bytes, float]:
    """One-shot synthesis (admin voice preview, latency bench): (PCM16 mono, seconds to first audio)."""
    key = api_key or os.getenv("ELEVENLABS_API_KEY", "")
    url = f"{WS_URL}?model_id={model}&output_format=pcm_{sample_rate}"
    if language:
        url += f"&language_code={language}"
    pcm = bytearray()
    t0 = time.monotonic()
    first = 0.0
    async with websockets.connect(url, additional_headers={"xi-api-key": key}, open_timeout=10, max_size=None) as ws:
        await ws.send(json.dumps({"voices": [voice_id], "voice_settings": {"stability": 0.5}}))
        await ws.send(json.dumps({"inputs": [{"text": text, "voice_id": voice_id, "new_turn": True}],
                                  "close_socket": True}))
        while True:
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                break
            if msg.get("error") or (msg.get("message") and msg.get("code")):
                raise RuntimeError(f"ElevenLabs: {msg.get('message') or msg.get('error')}")
            if msg.get("audio"):
                if not pcm:
                    first = time.monotonic() - t0
                pcm += base64.b64decode(msg["audio"])
            if msg.get("is_final"):
                break
    return bytes(pcm), first
