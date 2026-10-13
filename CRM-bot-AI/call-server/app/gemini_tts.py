"""Gemini 3.8 Flash / Flash-Lite TTS over the Interactions API (Gemini API key).

    POST /v1beta/interactions  stream=true  → SSE
      step.delta            delta.data = base64 headerless PCM16 mono at sample_rate
      interaction.completed interaction.usage.total_output_tokens = audio tokens (billed)

Language is auto-detected from the text (no locale field) — Genie's prompts already
force native script, which is what picks the accent. Voices are the same 30 names as
Chirp3-HD, so the agent's `voice` carries over.

TTFB was ~1.4 s (Lite) / ~1.9 s (Flash) per request on 25 Sep 2026 but 3.3-5 s on
8 Oct 2026 (from Fly sin, every variant; Chirp3-HD streaming: 0.2-0.3 s), so the
reply goes out as ONE request (WholeResponseAggregator): pipecat's default
per-sentence split would put a ~1.5 s hole after every sentence.
Ported from sahi-prompt-tester/gemini_audio.py.
"""

from __future__ import annotations

import base64
import json
import os
from typing import AsyncGenerator

import aiohttp
import numpy as np
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
MODELS = {"gemini_lite": "gemini-3.8-flash-lite-tts", "gemini_flash": "gemini-3.8-flash-tts"}


def _is_quiet(pcm: bytes, level: int = 300) -> bool:
    samples = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype=np.int16)
    return not samples.size or float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) < level


class WholeResponseAggregator(SimpleTextAggregator):
    """Buffer every LLM token; pipecat's TTSService flushes the whole reply on
    LLMFullResponseEndFrame, so the reply becomes a single TTS request."""

    async def aggregate(self, text: str):
        self._text += text
        return
        yield  # keep this an async generator


class GeminiTTS(TTSService):
    def __init__(self, *, session, api_key: str, aiohttp_session: aiohttp.ClientSession | None,
                 model: str, voice: str, style: str = "", chunking: str = "response",
                 sample_rate: int = 24000, **kwargs):
        super().__init__(sample_rate=sample_rate,
                         settings=TTSSettings(model=model, voice=voice, language=None), **kwargs)
        self._call = session            # CallSession: text clean-up + usage reporting
        self._api_key = api_key
        self._http_session = aiohttp_session
        self._own_http = False
        self._model = model
        self._voice = voice
        self._style = (style or "").strip()
        if chunking == "response":
            self._text_aggregator = WholeResponseAggregator()

    def can_generate_metrics(self) -> bool:
        return True

    def _body(self, text: str) -> dict:
        content = {"type": "text", "text": text}
        if self._style:
            content["annotations"] = [{"type": "speech_metadata", "style": self._style}]
        return {
            "model": self._model,
            "stream": True,
            "input": [{"type": "user_input", "content": [content]}],
            "response_format": {"type": "audio", "mime_type": "audio/l16",
                                "sample_rate": self.sample_rate},
            "generation_config": {"speech_config": [{"voice": self._voice}]},
        }

    async def _http(self) -> aiohttp.ClientSession:
        if self._http_session is None or self._http_session.closed:
            self._http_session, self._own_http = aiohttp.ClientSession(), True
        return self._http_session

    async def cleanup(self):
        await super().cleanup()
        if self._own_http and self._http_session and not self._http_session.closed:
            await self._http_session.close()

    @staticmethod
    async def _sse_payloads(resp: aiohttp.ClientResponse):
        """`data:` payloads as dicts. Buffers until a full line so a long base64 audio
        line is never cut in half by the transport's chunking."""
        buf = b""
        async for chunk in resp.content.iter_any():
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    logger.warning(f"Gemini TTS: bad SSE payload {payload[:80]!r}")

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        text = self._call.clean_tts_text(text)
        if not text:
            yield None
            return
        logger.debug(f"{self}: Gemini TTS [{text}]")
        headers = {"x-goog-api-key": self._api_key, "Content-Type": "application/json"}
        try:
            await self.start_ttfb_metrics()
            http = await self._http()
            async with http.post(INTERACTIONS_URL, json=self._body(text), headers=headers) as resp:
                if resp.status != 200:
                    body = (await resp.text())[:200]
                    logger.warning(f"Gemini TTS {resp.status}: {body}")
                    yield ErrorFrame(error=f"Gemini TTS {resp.status}: {body}")
                    return
                await self.start_tts_usage_metrics(text)
                got_audio = False
                # Gemini TTS sometimes keeps generating after the text is spoken (heard
                # on a phone call: ~35 s of babble + ~45 s of silence after a goodbye).
                # Expected length ≈ chars/9 s; stop at 2× that once the tail goes quiet,
                # and never exceed 2× + 6 s.
                expected = 2.0 + len(text) / 9
                hard_cap = 2 * expected + 6
                got_secs = quiet_secs = 0.0
                async for data in self._sse_payloads(resp):
                    # Keyed on payload shape, not the event name (the REST API says
                    # step.delta / interaction.completed, the SDK content.delta /
                    # interaction.complete).
                    delta = data.get("delta") or {}
                    if delta.get("data") and str(delta.get("mime_type", "")).startswith("audio"):
                        if not got_audio:
                            await self.stop_ttfb_metrics()
                            got_audio = True
                        pcm = base64.b64decode(delta["data"])
                        secs = len(pcm) / 2 / self.sample_rate
                        got_secs += secs
                        quiet_secs = quiet_secs + secs if _is_quiet(pcm) else 0.0
                        if got_secs > hard_cap or (got_secs > expected and quiet_secs >= 1.5):
                            logger.warning(f"Gemini TTS: cut at {got_secs:.1f}s for {len(text)} chars "
                                           f"(expected ~{expected:.0f}s, quiet tail {quiet_secs:.1f}s)")
                            break
                        yield TTSAudioRawFrame(pcm, self.sample_rate, 1, context_id=context_id)
                        continue
                    usage = (data.get("interaction") or {}).get("usage")
                    if usage:
                        self._call.add_tts_usage(int(usage.get("total_input_tokens") or 0),
                                                 int(usage.get("total_output_tokens") or 0))
                if not got_audio:
                    yield ErrorFrame(error="Gemini TTS: no audio in stream")
        except Exception as e:  # noqa: BLE001 — one failed reply must not end the call
            logger.warning(f"Gemini TTS failed: {e}")
            yield ErrorFrame(error=f"Gemini TTS failed: {e}")


async def synthesize(text: str, *, model: str, voice: str, style: str = "",
                     sample_rate: int = 24000) -> bytes:
    """One-shot synthesis (admin voice preview): the whole clip as PCM16 mono."""
    content = {"type": "text", "text": text}
    if style:
        content["annotations"] = [{"type": "speech_metadata", "style": style}]
    body = {"model": model, "stream": True,
            "input": [{"type": "user_input", "content": [content]}],
            "response_format": {"type": "audio", "mime_type": "audio/l16", "sample_rate": sample_rate},
            "generation_config": {"speech_config": [{"voice": voice}]}}
    pcm = bytearray()
    async with aiohttp.ClientSession() as http:
        async with http.post(INTERACTIONS_URL, json=body,
                             headers={"x-goog-api-key": os.getenv("GOOGLE_API_KEY", "")}) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Gemini TTS {resp.status}: {(await resp.text())[:200]}")
            async for data in GeminiTTS._sse_payloads(resp):
                delta = data.get("delta") or {}
                if delta.get("data") and str(delta.get("mime_type", "")).startswith("audio"):
                    pcm += base64.b64decode(delta["data"])
    return bytes(pcm)
