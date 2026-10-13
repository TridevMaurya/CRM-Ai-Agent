"""The voice pipeline for one call.

    transport.input → VAD → Sarvam saaras STT → TranscriptionGate → user turn →
    FillerPlayer ("hmm / achha" from cache) → Gemini LLM → TTS (Chirp3-HD, or
    Gemini 3.8 TTS per agent `tts.provider`) → transport.output

The transport is either Exotel (phone, 8 kHz PCM over WebSocket) or a browser
(protobuf frames — the /test page). Everything else is identical, which is the
point: the browser test call is the phone call.

Per call we also keep: the transcript (SQLite), a cost + latency meter, a hard
duration limit, and end-of-call detection (the agent ends its final line with
<END_CALL>).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time

import aiohttp
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMUsageMetricsData, TTSUsageMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.google.llm import GoogleLLMService, GoogleLLMSettings
from pipecat.services.google.tts import GoogleTTSService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport
from pipecat.turns.user_start.transcription_user_turn_start_strategy import (
    TranscriptionUserTurnStartStrategy,
)
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from . import store
from .fillers import FillerPlayer, PreRenderedSpeechFrame, SpeechWatcher, TurnEndTrigger, is_filler_echo
from .eleven_tts import ElevenTTS, DEFAULT_MODEL as ELEVEN_DEFAULT_MODEL
from .gemini_tts import MODELS as GEMINI_TTS_MODELS
TTS_PROVIDERS = {"chirp", "eleven", *GEMINI_TTS_MODELS}
from .gemini_tts import GeminiTTS
from .gemini_tts import synthesize as gemini_synthesize
from .agents import LANG_CODES, LANG_NAMES, render_prompt, resolve_language

INPUT_SAMPLE_RATE = 16000    # pipeline mic rate (Exotel's 8 kHz is resampled by the serializer)
OUTPUT_SAMPLE_RATE = 24000   # Chirp3-HD native; resampled to 8 kHz for the phone

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
STT_MODEL = os.getenv("STT_MODEL", "saaras:v3")

# Prices used by the per-call cost meter (override via env if they change).
RATE_CHIRP_USD_PER_1M_CHARS = float(os.getenv("RATE_CHIRP_USD_PER_1M_CHARS", "30"))
RATE_ELEVEN_USD_PER_1M_CHARS = float(os.getenv("RATE_ELEVEN_USD_PER_1M_CHARS", "11"))   # v4 Turbo, enterprise rate 8 Oct 2026
RATE_SARVAM_STT_INR_PER_HOUR = float(os.getenv("RATE_SARVAM_STT_INR_PER_HOUR", "30"))
RATE_LLM_USD_IN_PER_1M = float(os.getenv("RATE_LLM_USD_IN_PER_1M", "0.25"))
RATE_LLM_USD_OUT_PER_1M = float(os.getenv("RATE_LLM_USD_OUT_PER_1M", "1.50"))
# Gemini 3.8 TTS bills audio tokens (~40 per second of speech); prices till 31 Dec 2026.
RATE_GEMINI_TTS_USD_PER_1M_AUDIO_TOK = {
    "gemini_lite": float(os.getenv("RATE_GEMINI_LITE_TTS_USD_PER_1M_AUDIO_TOK", "6.0")),
    "gemini_flash": float(os.getenv("RATE_GEMINI_FLASH_TTS_USD_PER_1M_AUDIO_TOK", "9.0")),
}
RATE_GEMINI_TTS_USD_PER_1M_TEXT_TOK = float(os.getenv("RATE_GEMINI_TTS_USD_PER_1M_TEXT_TOK", "0.5"))
USD_INR = float(os.getenv("USD_INR", "88"))

END_TOKEN = "<END_CALL>"
_END_RE = re.compile(r"<\s*END_CALL\s*>", re.IGNORECASE)
# Step marker the agent prefixes every reply with ("§3 "): which CALL FLOW step it is
# on. Stripped before speaking / recording; used to remind the model of unasked steps.
# Tolerant of the model writing "§N 2" / "§N" (it copied the placeholder letter from
# the prompt on 13 Oct, and "§N" went to the TTS): the digit is optional, "N" is eaten.
_STEP_RE = re.compile(r"^\s*§\s*N?\s*(\d)")
_STEP_ANY_RE = re.compile(r"§\s*N?\s*\d?\s*[:.\-–—]?\s*")

# Chirp3-HD reads a lowercase standalone "us" as "U.S." — respell for TTS only.
_US_FIX = re.compile(r"\bus\b")
# Sheet titles are written for a thumbnail, not for speech: "Ep 1: Zero Investment…"
# came out as "App 1" / "AI", "MagicLights AI - Ab VIDEO banana bacho ka khel" was
# read in full with the dash. Strip the series prefix and tidy separators.
_TITLE_PREFIX = re.compile(r"^\s*(ep|episode|part|intro|day|lesson|class)\s*\.?\s*\d*\s*[:\-–—|]\s*", re.IGNORECASE)
_TITLE_SEP = re.compile(r"\s*[|\-–—]+\s*")
_EP_IN_TEXT = re.compile(r"\bEp\.?\s*(\d+)\b")


def speech_title(title: str) -> str:
    t = _TITLE_PREFIX.sub("", (title or "").strip())
    t = _TITLE_SEP.sub(", ", t)
    t = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", t)   # emoji
    return re.sub(r"\s+", " ", t).strip(" ,") or (title or "").strip()
_PUNCT_ONLY = re.compile(r"^[\s.…,;:!?\-–—\"'“”‘’()\[\]<>]*$")

# The script of each call language. A transcript in some other Indian script
# ("WC ആണി" in a Telugu call, Bengali letters in a Hindi call) is Sarvam guessing on
# noise or on Genie's own echo — never the caller. Latin (English words) is always ok.
_SCRIPT_OF = {
    "hin": r"ऀ-ॿ", "mar": r"ऀ-ॿ", "tel": r"ఀ-౿", "tam": r"஀-௿", "kan": r"ಀ-೿",
    "mal": r"ഀ-ൿ", "ben": r"ঀ-৿", "eng": r"",
}


def _is_foreign_garbage(text: str, lang: str = "") -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    allowed = re.compile(f"[a-zA-ZÀ-ɏ{_SCRIPT_OF.get(lang, 'ऀ-ॿ஀-௿ఀ-౿ಀ-೿ഀ-ൿঀ-৿')}]")
    return sum(1 for c in letters if allowed.match(c)) / len(letters) < 0.6


# "Hang up" in any words: after Genie's next line the call ends, whether or not the
# model remembered <END_CALL>.
_REFUSAL = re.compile(r"((phone|फोन)\s*(rakh|rakho|rakhiye|kaat|cut|रख|रखो|रखिए|काट)|(call|कॉल)\s*(kaat|cut|mat|band|काट|मत|बंद)|"
                      r"cut the call|band karo|बंद करो|मत करो|नहीं चाहिए|nahi chahiye|disturb mat|परेशान मत|रखो फोन|রাখ)", re.IGNORECASE)

# Money back, asked a SECOND time: Genie can't refund, and soothing for another
# minute only made calls worse (12 Oct QC: "मेरा पैसा return हो सकता है?" ×6, the same
# canned line back each time). Second ask → one line + end.
_REFUND = re.compile(r"((पैस|रुपय|रुपए|₹|paisa|paise|money|रिफंड|refund).{0,25}(वापस|return|लौटा|wapas|refund|रिफंड))|"
                     r"((वापस|return|refund|रिफंड|लौटा).{0,25}(पैस|रुपय|paisa|paise|money))", re.IGNORECASE)
# "Call later" / busy / the app's user isn't the one on the line (a child, a spouse).
# The prompt had the rule; the model kept asking anyway ("मेरी मम्मी अभी आई नहीं, वो
# आएँगी तब पूछ लेना" → "बहुत बढ़िया, अगर आप वीडियो बनाना सीखना चाहती हैं…").
_CALLBACK = re.compile(r"((बाद में|थोड़ी देर|baad me|thodi der).{0,15}(call|कॉल|फोन|phone|बात|कर लेना|करना|करो|कीजिए|करिए))|"
                       r"busy|व्यस्त|काम (कर|पर|पे) (रह|हूँ|हूं|है)|बजे.{0,20}(बात|call|कॉल|फोन)|(टाइम|समय|time) नहीं|"
                       r"फ्री नहीं|गाड़ी चला|driving|ड्राइव|meeting|मीटिंग|hospital|अस्पताल|तबीयत|बीमार हूँ|बीमार हूं", re.IGNORECASE)
_WRONG_PERSON = re.compile(r"((मम्मी|पापा|भैया|दीदी|पति|बीवी|बेटा|बेटे|भाई|उनका|उनके|उनकी).{0,25}"
                           r"(नहीं (है|हैं)|(आई|आए|आये) नहीं|गई|गए|बाहर|घर (पे|पर) नहीं))|wrong number|गलत नंबर|"
                           r"पढ़(ा|े)[\s-]*लिख(ा|े) नहीं|(बच्चे|बच्चा|बच्चों).{0,12}(चलात|देखत|install|डाउनलोड)|"
                           r"(हमारा|मेरा) (फोन|phone|नंबर) नहीं|हिंदी नहीं (आती|समझ)|(बंगाली|बांग्ला|तेलुगु|तमिल) में", re.IGNORECASE)
# Abuse: the model kept treating slurs as feedback ("aap kehna chah rahe hain ki video
# tez the?"). One line and out.
_ABUSE = re.compile(r"भोसड़|भोसड|मादरचोद|मादरचो|बहनचोद|बहन के|चूतिय|चुतिय|गांड|गाण्ड|लौड़|लंड\b|रंडी|हरामी|कमीन|साले\b|साला\b|"
                    r"madarch|behench|bhosd|chutiy|gaand|lod[ue]|randi|harami|kamin", re.IGNORECASE)

# A bare greeting from the caller ("hello", "haan", "ji") — not an answer.
_BARE_HELLO = re.compile(
    r"^[\s.,!?।]*((hello|hallo|helo|haan|haa|han|ji|hanji|haanji|bolo|boliye|bataiye|batao|"
    r"हेलो|हैलो|हलो|हाँ|हां|जी|बोलिए|बोलो|बताइए|बताइये|बताओ|सुन रहा हूँ|सुन रहा हूं|"
    r"சொல்லுங்க|చెప్పండి|హలో|ஹலோ|ಹಲೋ|ഹലോ)[\s.,!?।]*){1,3}$", re.IGNORECASE)


class TranscriptionGate(FrameProcessor):
    """Drop STT noise (foreign-script hallucinations, empty turns), and decide what
    happens to speech while Genie is talking:

    - barge-in on (`interruptions.min_words` > 0): N+ words while Genie speaks
      interrupt her (her audio stops) and the words become the user's turn. Fewer
      words — echo, "hmm", a cough — are ignored and she carries on.
    - barge-in off: everything said while she speaks is ignored (echo protection).

    Also drops what the STT hears while a turn-end filler plays, a bare "hmm/achha"
    right after a backchannel (handset echo), and records user activity for the
    silence hang-up."""

    def __init__(self, session: "CallSession", interrupt_words: int = 0, lang: str = ""):
        super().__init__()
        self._session = session
        self._interrupt_words = interrupt_words
        self._lang = lang
        self._bot_speaking = False
        self._hellos = 0

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = self._session.bot_speaking = self._session.bot_spoke = True
            self._session.bot_started_at = time.monotonic()
            if not self._session.nudging:
                self._session.touch()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = self._session.bot_speaking = False
            if self._session.filler_playing:
                # A hum / backchannel just ended, not Genie's reply: the reply to the
                # last turn is still pending, so a new turn must still cancel it.
                self._session.filler_playing = False
            else:
                self._session.awaiting_reply = False
            if self._session.nudging:
                self._session.nudging = False      # a nudge is not a new question
            else:
                self._session.touch()
        elif isinstance(frame, (VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)):
            self._session.last_vad_at = time.monotonic()
            self._session.user_speaking = isinstance(frame, VADUserStartedSpeakingFrame)
        if isinstance(frame, TranscriptionFrame):
            text = (frame.text or "").strip()
            if not text or _is_foreign_garbage(text, self._lang):
                logger.info(f"[gate] dropped: {text!r}")
                return
            # People say "hello" as they pick up, before Genie has said a word. That
            # is not a turn — with a reply it made Genie greet twice, back to back.
            if not self._session.bot_spoke and time.monotonic() - self._session.started < 8:
                logger.info(f"[gate] dropped pre-greeting: {text!r}")
                return
            self._session.touch()
            self._session.user_turns += 1
            if self._bot_speaking:
                words = len(text.split())
                if self._interrupt_words and words >= self._interrupt_words and not is_filler_echo(text):
                    logger.info(f"[gate] INTERRUPT ({words}w): {text!r}")
                    await self.broadcast_interruption()
                    await self.push_frame(frame, direction)
                    return
                logger.info(f"[gate] ignored while speaking: {text!r}")
                return
            if time.monotonic() < self._session.filler_until:
                logger.info(f"[gate] ignored during filler: {text!r}")
                return
            if time.monotonic() < self._session.backchannel_until and is_filler_echo(text):
                logger.info(f"[gate] dropped backchannel echo: {text!r}")
                return
            if _BARE_HELLO.match(text):
                # "Hello?" after the greeting = "I'm listening" (or "I didn't hear you").
                # Hand the model an explicit instruction instead of a bare word it keeps
                # answering with the whole opening again.
                self._hellos += 1
                self._session.record("user", text)      # transcript keeps what they said
                frame.recorded = True                   # …so the recorder skips the note
                frame.text = ("(The user said \"hello / haan boliye\" — they are listening. Do NOT "
                              "introduce yourself again and do not repeat the opening. Ask ONE concrete "
                              "question about the video they watched — name what it showed (from its "
                              "title) and ask if it made sense / if they tried it. Under 15 words, in "
                              "the call language's own script, no Latin-script Hinglish.)"
                              if self._hellos == 1 else
                              "(The user said \"hello\" again — they may not have heard you. Repeat "
                              "your current question in under 10 words, in the call language's own "
                              "script. Nothing else — no \"main sun rahi hoon\".)")
                logger.info(f"[gate] bare hello #{self._hellos}: {text!r}")
            if _REFUSAL.search(text):
                logger.info(f"[gate] refusal — ending after Genie's next line: {text!r}")
                self._session.end_after_reply = True
            elif _REFUND.search(text):
                self._session.refund_asks += 1
                if self._session.refund_asks >= 2 and not getattr(frame, "recorded", False):
                    logger.info(f"[gate] refund asked again — ending after Genie's next line: {text!r}")
                    self._session.record("user", text)
                    frame.recorded = True
                    frame.text = (text + "\n(They are asking for the money back AGAIN. Reply in ONE short line in "
                                  "the call language's script: refund is not in your hands, their payment "
                                  "complaint is noted for the team — thank them and end with <END_CALL>. "
                                  "No question, no explanation of what Gyan TV is.)")
                    self._session.end_after_reply = True
            elif _ABUSE.search(text) and not getattr(frame, "recorded", False):
                logger.info(f"[gate] abuse — ending after Genie's next line")
                self._session.record("user", text)
                frame.recorded = True
                frame.text = ("(The user is abusive. Reply with ONLY \"ठीक है, शुक्रिया।\" and <END_CALL>. "
                              "Do not answer, explain, or treat it as feedback.)")
                self._session.end_after_reply = True
            elif (_CALLBACK.search(text) or _WRONG_PERSON.search(text)) and not getattr(frame, "recorded", False):
                logger.info(f"[gate] busy / call later / wrong person — ending after Genie's next line: {text!r}")
                self._session.record("user", text)
                frame.recorded = True
                frame.text = (text + "\n(They are busy, asked to be called later, or the app's user is not the "
                              "person on the line. Reply in ONE short line in the call language's script: "
                              "theek hai, main baad mein call kar lungi, shukriya — and end with "
                              "<END_CALL>. No question.)")
                self._session.end_after_reply = True
            missing = [f"{n} ({label})" for n, label in sorted(self._session.must_ask.items())
                       if n not in self._session.steps_asked]
            if missing and self._session.user_turns >= 3 and not getattr(frame, "recorded", False):
                # The model drifts to the closing after one good answer; keep the unasked
                # mandatory steps in front of it (never spoken, not in the transcript).
                self._session.record("user", text)
                frame.recorded = True
                frame.text = (text + "\n(Reminder — CALL FLOW steps not asked yet: " + ", ".join(missing)
                              + ". Ask them, one at a time, before any closing.)")
                logger.info(f"[gate] reminder: steps missing {missing}")
            if self._session.awaiting_reply:
                # A second user turn while the previous reply is still being generated or
                # spoken: cancel that reply so the user gets ONE answer to both, not two.
                logger.info(f"[gate] new turn while a reply is pending — cancelling it")
                await self.broadcast_interruption()
            self._session.awaiting_reply = True
        await self.push_frame(frame, direction)


# Words a Hindi/Hinglish speaker ends a *clause* on, not a thought: "मतलब", "और",
# "जैसे कि"… If the transcript so far ends on one of these (or a comma / trailing
# dots) the user is still going and Genie must wait longer before replying.
_CONTINUATION = {
    "और", "मतलब", "तो", "जैसे", "कि", "लेकिन", "फिर", "क्योंकि", "जो", "पर", "या", "अगर", "जब",
    "मेरा", "मेरे", "मेरी", "मैं", "हम", "वो", "ये", "यह", "वह", "एक", "थोड़ा", "कुछ", "उसमें", "इसमें",
    "बस", "अभी", "क्या", "जिसमें", "उसका", "उसके", "इसका", "इसके", "मुझे", "हमें", "आपका", "आपके",
    "aur", "matlab", "toh", "to", "jaise", "ki", "lekin", "phir", "kyunki", "jo", "par", "ya", "agar",
    "jab", "mera", "mere", "meri", "main", "hum", "wo", "ye", "ek", "thoda", "kuch", "usme", "isme",
    "bas", "abhi", "kya", "mujhe", "hume", "aapka", "aapke", "and", "but", "so", "because", "like",
}
_TRAIL_PUNCT = re.compile(r"[।.!?\s]+$")


def looks_unfinished(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if t.endswith(("...", "…", ",", "،")):
        return True
    t = _TRAIL_PUNCT.sub("", t)
    last = t.split()[-1].lower().strip("\"'") if t.split() else ""
    return last in _CONTINUATION


class UnfinishedAwareTurnStop(SpeechTimeoutUserTurnStopStrategy):
    """Speech-timeout turn stop whose wait depends on what has been said: `base`
    seconds after a finished-sounding sentence, `hold` seconds when the words so far
    look mid-thought. The timer is re-armed with the longer wait if it is already
    running (or has even elapsed) when such a transcript arrives."""

    def __init__(self, *, base: float, hold: float, **kw):
        super().__init__(user_speech_timeout=base, **kw)
        self._base, self._hold = base, max(hold, base)

    async def reset(self):
        await super().reset()
        self._user_speech_timeout = self._base

    async def _handle_transcription(self, frame):
        want = self._hold if looks_unfinished(self._text + (frame.text or "")) else self._base
        prev, self._user_speech_timeout = self._user_speech_timeout, want
        if want > prev and (self._user_speech_timeout_task or self._user_speech_wait_done):
            await self._restart_user_speech_timer()
        await super()._handle_transcription(frame)


class UserTurnRecorder(FrameProcessor):
    """Sits after the gate: writes each user turn to the store."""

    def __init__(self, session: "CallSession"):
        super().__init__()
        self._session = session

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and (frame.text or "").strip():
            if not getattr(frame, "recorded", False):
                self._session.record("user", frame.text.strip())
            self._session.user_spoke = True
        await self.push_frame(frame, direction)


_SENT_END = re.compile(r"(?<=[?।.!])\s+")


def _question_of(text: str) -> str:
    """The question part of a reply: every sentence with a '?' (else the last one).
    "…बंद क्यों किया? क्या कोई खास वजह थी?" is one question about the cancellation."""
    parts = [x.strip() for x in _SENT_END.split(text.strip()) if x.strip()]
    if not parts:
        return text.strip()
    qs = [x for x in parts if "?" in x]
    return " ".join(qs) if qs else parts[-1]


# Same idea, different spelling — collapse before comparing questions.
_SYNONYMS = {
    "video": "वीडियो", "videos": "वीडियो", "वीडियोज़": "वीडियो", "वीडियोस": "वीडियो", "वीडियोज": "वीडियो",
    "content": "कंटेंट", "story": "कहानी", "स्टोरी": "कहानी", "topic": "टॉपिक", "topics": "टॉपिक",
    "टॉपिक्स": "टॉपिक", "app": "ऐप", "apps": "ऐप", "एप": "ऐप", "voice": "आवाज़", "आवाज": "आवाज़",
    "visuals": "विजुअल्स", "विज़ुअल्स": "विजुअल्स", "category": "कैटेगरी", "categories": "कैटेगरी",
    "कैटेगरीज़": "कैटेगरी", "subscription": "सब्सक्रिप्शन", "trial": "ट्रायल", "price": "कीमत",
    "दाम": "कीमत", "feedback": "फीडबैक", "gyan": "gyantv", "tv": "", "ज्ञान": "gyantv", "टीवी": "",
}
# Scaffolding words every question shares — ignored so only the topic words count.
_STOP = set("""ये यह वो वह तो और या के का की को में पर से है हैं था थी थे हो हूँ हूं क्या कैसा कैसी कैसे कौन कौनसा कौनसी
कितना कितनी बताइए बताइये बताओ बता बताएं सकते सकती सकता चाहेंगे चाहेंगी चाहते चाहती लगा लगी लगे लगता लगती
आपको आपका आपकी आपके आप हम हमारे मुझे मैं अब अच्छा अच्छी अच्छे बहुत थोड़ा सबसे कुछ कोई एक जो जिस उस इस
सोचना सोचते सोचती राय मतलब यानी वैसे ज़रा जरा फिर बारे लिए साथ ही भी ना नहीं जी हाँ हां ok okay ji
आपने किया किए की करने कर सके होनी चाहिए अगर आगे किस तरह करेंगे करेंगी क्यों मुख्य खास
ख्याल उसे उसमें वीडियो gyantv आखिरी सवाल जाते बस दीजिए बताइये पहले हम
what which how do you the a an is are was were about your in on of for to and or""".split())


_SUFFIXES = ("ेंगे", "ेंगी", "ना", "ने", "नी", "ता", "ती", "ते", "ें", "ों", "ी", "े", "ा", "ो")


def _stem(w: str) -> str:
    """Crude Hindi stem so सुधारें / सुधारने / सुधार count as one word."""
    for suf in _SUFFIXES:
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _topic_words(question: str) -> set[str]:
    words = re.findall(r"[\w\u0900-\u097F]+", question.lower())
    out = set()
    for w in words:
        w = _SYNONYMS.get(w, w)
        if w and w not in _STOP and len(w) > 1:
            out.add(_stem(w))
    return out


def same_question(a: str, b: str) -> bool:
    """Two questions about the same thing? Judged on their topic words."""
    ta, tb = _topic_words(a), _topic_words(b)
    if not ta or not tb:
        return False
    small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    overlap = len(ta & tb)
    return overlap >= 2 and (overlap / len(small) >= 0.6 or small <= big)


# Scripted openings: spoken the moment the call connects (straight to the TTS, no LLM
# round trip ≈ 1.5 s saved). The line is added to the context as Genie's first turn so
# the model continues from it. Other languages / unknown cohorts use the LLM opening.
GREETINGS = {
    "hin": {
        "video": "नमस्ते, मैं Gyan TV से Genie बोल रही हूँ। आपने हाल ही में '{title}' वाला video देखा था — वो कैसा लगा, दो मिनट में बता सकते हैं?",
        "video_untitled": "नमस्ते, मैं Gyan TV से Genie बोल रही हूँ। आपने हाल ही में Gyan TV पे कुछ videos देखे थे — वो कैसे लगे, दो मिनट में बता सकते हैं?",
        "TRIAL_CANCELLED": "नमस्ते, मैं Gyan TV से Genie बोल रही हूँ। आपने हाल ही में अपना trial cancel किया था — क्यों, दो मिनट में बता सकते हैं?",
        "SUB_CANCELLED": "नमस्ते, मैं Gyan TV से Genie बोल रही हूँ। आपने हाल ही में अपना subscription cancel किया था — क्यों, दो मिनट में बता सकते हैं?",
        # incoming: they called us back after a missed call — they don't know who we
        # are ("कौन था?", "कौन सा?"), so say that first.
        "in_video": "नमस्ते, मैं Gyan TV से Genie बोल रही हूँ — आपको हमारी तरफ से call आया था। आपने हाल ही में '{title}' वाला video देखा था — वो कैसा लगा, दो मिनट में बता सकते हैं?",
        "in_video_untitled": "नमस्ते, मैं Gyan TV से Genie बोल रही हूँ — आपको हमारी तरफ से call आया था। आपने हाल ही में Gyan TV पे कुछ videos देखे थे — वो कैसे लगे, दो मिनट में बता सकते हैं?",
        "in_TRIAL_CANCELLED": "नमस्ते, मैं Gyan TV से Genie बोल रही हूँ — आपको हमारी तरफ से call आया था। आपने हाल ही में अपना trial cancel किया था — क्यों, दो मिनट में बता सकते हैं?",
        "in_SUB_CANCELLED": "नमस्ते, मैं Gyan TV से Genie बोल रही हूँ — आपको हमारी तरफ से call आया था। आपने हाल ही में अपना subscription cancel किया था — क्यों, दो मिनट में बता सकते हैं?",
        "in_unknown": "नमस्ते, मैं Gyan TV से Genie बोल रही हूँ — आपको हमारी तरफ से call आया था। बताइए, आप Gyan TV पर कौन सा video देख रहे थे?",
    },
}


# phone → (greeting text, pcm16 24 kHz) rendered at dial time; used once on connect.
_PREGEN: dict[str, tuple[str, bytes, float]] = {}
PREGEN_TTL = 600


async def pregen_greeting(number: str, agent: dict, params: dict) -> None:
    """Synthesise the scripted opening while Exotel is still ringing the user, so it
    plays the instant the call connects (Gemini TTS alone needs 1.4-5 s; Chirp ~0.3 s, so chirp skips this)."""
    from .agents import resolve_language
    from . import store as _store
    lang = resolve_language(agent, params)
    text = scripted_greeting(lang, {**params, "direction": "outbound"})
    provider = str(params.get("tts") or (agent.get("tts") or {}).get("provider") or "chirp")
    if not text or provider not in GEMINI_TTS_MODELS:
        return
    try:
        pcm = await gemini_synthesize(text, model=GEMINI_TTS_MODELS[provider],
                                      voice=params.get("voice") or agent["voice"],
                                      style=str(params.get("tts_style") or (agent.get("tts") or {}).get("style") or ""),
                                      sample_rate=OUTPUT_SAMPLE_RATE)
        if pcm:
            _PREGEN[_store.normalize_number(number)] = (text, pcm, time.monotonic())
            logger.info(f"[pregen] greeting ready for …{number[-4:]} ({len(pcm) / 2 / OUTPUT_SAMPLE_RATE:.1f}s)")
    except Exception as e:  # noqa: BLE001 — falls back to live TTS on connect
        logger.warning(f"[pregen] failed: {e}")


# Fixed opening lines rendered once per (text, model, voice, style) and kept for the
# life of the process: inbound calls (nothing is pre-rendered at dial time) and
# outbound calls answered before the per-number pregen finished used to wait 3–12 s
# for live Gemini TTS before hearing a word. Title-specific greetings can't be
# cached whole, so their fixed first sentence (INTRO) plays from cache while the
# rest goes to TTS.
_CLIPS: dict[tuple[str, str, str, str], bytes] = {}
_CLIPS_PENDING: set[tuple[str, str, str, str]] = set()
INTRO = {
    "hin": {"outbound": "नमस्ते, मैं Gyan TV से Genie बोल रही हूँ।",
            "inbound": "नमस्ते, Gyan TV पर call करने के लिए शुक्रिया, मैं Genie बोल रही हूँ।"},
}


def cached_clip(text: str, provider: str, voice: str, style: str) -> bytes | None:
    """The clip if rendered; otherwise start rendering it for next time and return None."""
    key = (text, provider, voice, style)
    hit = _CLIPS.get(key)
    if hit is None and key not in _CLIPS_PENDING and provider in GEMINI_TTS_MODELS:
        _CLIPS_PENDING.add(key)
        asyncio.create_task(_render_clip(key))
    return hit


async def _render_clip(key: tuple[str, str, str, str]) -> None:
    text, provider, voice, style = key
    try:
        pcm = await gemini_synthesize(text, model=GEMINI_TTS_MODELS[provider], voice=voice,
                                      style=style, sample_rate=OUTPUT_SAMPLE_RATE)
        if pcm:
            _CLIPS[key] = pcm
            logger.info(f"[clips] cached {text[:40]!r}… ({len(pcm) / 2 / OUTPUT_SAMPLE_RATE:.1f}s, {voice})")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[clips] render failed for {text[:40]!r}: {e}")
    finally:
        _CLIPS_PENDING.discard(key)


def tts_identity(agent: dict, params: dict) -> tuple[str, str, str]:
    """(provider, voice, style) a call will speak with — same resolution as run()."""
    tts_cfg = agent.get("tts") or {}
    provider = str(params.get("tts") or tts_cfg.get("provider") or "chirp").strip().lower()
    return (provider, str(params.get("voice") or agent["voice"]),
            str(params.get("tts_style") or tts_cfg.get("style") or ""))


async def warm_clips(agents: list[dict]) -> None:
    """Render every fixed greeting and intro for each agent's current voice, in the
    background at start-up (and again whenever admin changes the voice — the first
    call after a change fills the cache lazily)."""
    for agent in agents:
        provider, voice, style = tts_identity(agent, {})
        if provider not in GEMINI_TTS_MODELS:
            continue
        for lang, g in GREETINGS.items():
            texts = [t for t in g.values() if "{title}" not in t] + list(INTRO.get(lang, {}).values())
            for text in texts:
                cached_clip(text, provider, voice, style)
                await asyncio.sleep(0.3)      # don't burst the TTS API


def take_pregen(number: str) -> tuple[str, bytes] | None:
    from . import store as _store
    key = _store.normalize_number(number)
    hit = _PREGEN.pop(key, None)
    for k in [k for k, v in _PREGEN.items() if time.monotonic() - v[2] > PREGEN_TTL]:
        _PREGEN.pop(k, None)
    return (hit[0], hit[1]) if hit and time.monotonic() - hit[2] <= PREGEN_TTL else None


def scripted_greeting(lang: str, params: dict) -> str | None:
    g = GREETINGS.get(lang)
    if not g:
        return None
    inbound = str(params.get("direction", "")).lower() == "inbound"
    pre = "in_" if inbound else ""
    cohort = str(params.get("cohort", "")).upper()
    title = str(params.get("video_title") or "").strip()
    if cohort in ("TRIAL_CANCELLED", "SUB_CANCELLED"):
        return g[pre + cohort]
    if cohort in ("LIGHT_3_4", "DAU_10PLUS") or title:
        return g[pre + "video"].format(title=speech_title(title)) if title else g[pre + "video_untitled"]
    return g["in_unknown"] if inbound else None
    cohort = str(params.get("cohort", "")).upper()
    if cohort in g:
        return g[cohort]
    if cohort in ("LIGHT_3_4", "DAU_10PLUS") or params.get("video_title"):
        title = str(params.get("video_title") or "").strip()
        return g["video"].format(title=title) if title else g["video_untitled"]
    return None


GOODBYE = {
    "hin": "ठीक है, समझ गई। आपके समय के लिए बहुत-बहुत शुक्रिया, नमस्ते। <END_CALL>",
    "mar": "ठीक आहे, समजलं. तुमच्या वेळेसाठी खूप खूप धन्यवाद, नमस्कार. <END_CALL>",
    "tel": "సరే, అర్థమైంది. మీ సమయానికి చాలా ధన్యవాదాలు, నమస్తే. <END_CALL>",
    "tam": "சரி, புரிஞ்சுது. உங்க நேரத்துக்கு ரொம்ப நன்றி, வணக்கம். <END_CALL>",
    "kan": "ಸರಿ, ಅರ್ಥವಾಯ್ತು. ನಿಮ್ಮ ಸಮಯಕ್ಕೆ ತುಂಬಾ ಧನ್ಯವಾದಗಳು, ನಮಸ್ಕಾರ. <END_CALL>",
    "mal": "ശരി, മനസ്സിലായി. നിങ്ങളുടെ സമയത്തിന് വളരെ നന്ദി, നമസ്കാരം. <END_CALL>",
    "eng": "Okay, got it. Thank you so much for your time, goodbye. <END_CALL>",
}

# Said after `nudge_after_secs` of silence following Genie's question.
NUDGE = {
    "hin": "Hello? आप सुन रहे हैं?", "mar": "Hello? तुम्ही ऐकताय का?", "tel": "Hello? మీరు వింటున్నారా?",
    "tam": "Hello? நீங்க கேக்குறீங்களா?", "kan": "Hello? ನೀವು ಕೇಳ್ತಾ ಇದ್ದೀರಾ?",
    "mal": "Hello? നിങ്ങൾ കേൾക്കുന്നുണ്ടോ?", "eng": "Hello? Are you there?",
}

# When the model insists on re-asking, Genie jumps to the closing question instead.
REPEAT_FALLBACK = {
    "hin": "ठीक है, समझ गई। आखिरी सवाल — अगर Gyan TV आपके लिए एक चीज़ बेहतर कर सके, तो वो क्या होनी चाहिए?",
    "mar": "ठीक आहे, समजलं. शेवटचा प्रश्न — Gyan TV तुमच्यासाठी एक गोष्ट सुधारू शकलं तर ती काय असावी?",
    "tel": "సరే, అర్థమైంది. చివరి ప్రశ్న — Gyan TV మీ కోసం ఒక్క విషయం మెరుగుపరచాలంటే అది ఏమిటి?",
    "tam": "சரி, புரிஞ்சுது. கடைசி கேள்வி — Gyan TV உங்களுக்காக ஒரு விஷயத்தை மேம்படுத்தணும்னா அது என்ன?",
    "kan": "ಸರಿ, ಅರ್ಥವಾಯ್ತು. ಕೊನೆಯ ಪ್ರಶ್ನೆ — Gyan TV ನಿಮಗಾಗಿ ಒಂದು ವಿಷಯ ಸುಧಾರಿಸಬೇಕಾದರೆ ಅದು ಏನು?",
    "mal": "ശരി, മനസ്സിലായി. അവസാന ചോദ്യം — Gyan TV നിങ്ങൾക്കായി ഒരു കാര്യം മെച്ചപ്പെടുത്തണമെങ്കിൽ അത് എന്താണ്?",
    "eng": "Okay, got it. Last question — if Gyan TV could improve one thing for you, what should it be?",
}


# Stock openers the model puts on almost every reply ("समझ गई, …" opened 27 % of all
# bot turns, some stock acknowledgement 45 %, 12 Oct QC of 142 calls). Each is allowed
# ONCE per call; after that the reply starts with its substance.
_ACK_OPENER = re.compile(
    r"^(?P<step>\s*§\s*\d\s*)?(?P<hum>(हम्म+|हाँ|हां|जी)[,।!.\s]+)?"
    r"(?P<ack>समझ गई|मैं (आपकी (बात|परेशानी|चिंता|नाराज़गी|नाराजगी) )?(पूरी तरह )?समझ (गई|सकती)( हूँ)?|ठीक है|अच्छा|"
    r"कोई बात नहीं|जी, मैं सुन रही हूँ|बिल्कुल|बहुत बढ़िया|चलिए)[,।!.\s]+")
# A closing ("…shukriya", "aapka din shubh ho") glued after a question: the model asks
# the last question and says goodbye in the same breath, so the user never answers.
_GOODBYE_SENT = re.compile(r"शुक्रिया|धन्यवाद|thank you|दिन शुभ|ख्याल रख|नमस्ते", re.IGNORECASE)
# A closing line the model forgot to end with <END_CALL>: thanks for their time / their
# opinion and no question. Without this the call ran on after the goodbye (37 of 142
# calls, 12 Oct QC), Genie answering background noise until the user hung up.
_GOODBYE_LINE = re.compile(r"((समय|टाइम|time|वक्त|राय|फीडबैक|feedback|बात|जानकारी).{0,50}(शुक्रिया|धन्यवाद|thank))|"
                           r"((शुक्रिया|धन्यवाद|thank).{0,50}(समय|टाइम|time|वक्त|दिन))|दिन शुभ|ख्याल रख|नमस्ते[।!.\s]*$",
                           re.IGNORECASE)


def tidy_reply(text: str, session: "CallSession") -> str:
    """Drop a stock opener that was already used once this call, and a goodbye glued
    after a question. Returns the text to speak (unchanged when nothing applies)."""
    out = text
    m = _ACK_OPENER.match(out)
    if m:
        key = re.sub(r"\s+", " ", m.group("ack")).strip()
        rest = out[m.end():].strip()
        if key.startswith("जी, मैं सुन") or key in session.openers_used:
            if len(rest) >= 4:
                out = (m.group("step") or "") + rest
        else:
            session.openers_used.add(key)
    parts = [p for p in _SENT_END.split(out.strip()) if p]
    last_q = max((i for i, p in enumerate(parts) if "?" in p), default=-1)
    if 0 <= last_q < len(parts) - 1:
        tail = parts[last_q + 1:]
        if all(_GOODBYE_SENT.search(p) or _END_RE.search(p) or len(p) < 4 for p in tail):
            out = " ".join(parts[:last_q + 1])
    return out


class ReplyGuard(FrameProcessor):
    """Sits right after the LLM. Holds each reply until it is complete, and if its
    question is a near-repeat of one Genie already asked in this call, drops it and
    asks the model again with a note (once per turn). The model keeps re-asking
    "story kaisi lagi?" in other words when an answer only half covers it — the
    prompt rules alone didn't stop that. Also tidies the reply (see tidy_reply)."""

    def __init__(self, session: "CallSession", lang: str):
        super().__init__()
        self._session = session
        self._lang = lang
        self._buf: list = []
        self._holding = False
        self._retry_pending = False   # the current response is the model's second try

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            self._buf, self._holding = [], False
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._buf, self._holding = [frame], True
            return
        elif self._holding and isinstance(frame, LLMTextFrame):
            self._buf.append(frame)
            return
        elif self._holding and isinstance(frame, LLMFullResponseEndFrame):
            self._holding = False
            text = "".join(f.text for f in self._buf if isinstance(f, LLMTextFrame))
            q = _question_of(_END_RE.sub("", text))
            prev = [_question_of(t) for t in self._session.bot_texts[-8:]]
            hits = [p for p in prev if p and "?" in p and same_question(q, p)]
            # Asking once more after a non-answer ("awaaz aa rahi hai?") is fine; the
            # THIRD time is the loop we stop.
            hit = hits[-1] if len(hits) >= 2 else None
            repeat = "?" in q and len(q) > 12 and hit is not None
            best = 1.0 if repeat else 0.0
            if repeat and not self._retry_pending:
                # First time: give the model one chance to move on by itself.
                self._retry_pending = True
                logger.info(f"[{self._session.call_id}] repeat guard: {q!r} ≈ {hit!r} — re-asking")
                self._buf = []
                await self.push_frame(LLMMessagesAppendFrame(messages=[{
                    "role": "user",
                    "content": "(You were about to ask a question you already asked in this call. "
                               "Do NOT ask it again in any wording. Treat it as answered, react in "
                               "at most 4 words, and ask the NEXT question on your list — or, if "
                               "none is left, say goodbye with <END_CALL>.)"}], run_llm=True))
                return
            if repeat and self._lang in REPEAT_FALLBACK:
                # Still repeating: Genie moves to the closing question herself — or, if
                # the closing question is what's being repeated, says goodbye.
                closing = same_question(q, _question_of(REPEAT_FALLBACK[self._lang]))
                logger.info(f"[{self._session.call_id}] repeat guard: {q!r} ≈ {hit!r} — model insisted, "
                            f"{'goodbye' if closing else 'fallback'}")
                self._buf = [self._buf[0], LLMTextFrame(text=GOODBYE[self._lang] if closing
                                                        else REPEAT_FALLBACK[self._lang])]
            self._retry_pending = False
            spoken = "".join(f.text for f in self._buf if isinstance(f, LLMTextFrame))
            tidy = tidy_reply(spoken, self._session)
            if tidy != spoken:
                logger.info(f"[{self._session.call_id}] tidy: {spoken!r} → {tidy!r}")
                self._buf = [self._buf[0], LLMTextFrame(text=tidy)]
            for f in self._buf:
                await self.push_frame(f, direction)
            self._buf = []
            await self.push_frame(frame, direction)
            return
        await self.push_frame(frame, direction)


class BotTurnRecorder(FrameProcessor):
    """Sits after the LLM: collects the streamed response, writes it as one bot
    turn, and decides whether this response ends the call. The agent is told to
    end its closing line with <END_CALL>; models sometimes append it to a
    question, so the token only counts when the response isn't asking anything."""

    def __init__(self, session: "CallSession"):
        super().__init__()
        self._session = session
        self._buf: list[str] = []

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            # The user cut Genie off — whatever she was saying (even a goodbye) is
            # abandoned, so the call goes on.
            if self._buf:
                self._session.record("bot", "".join(self._buf).strip() + " …[interrupted]")
            self._buf = []
            self._session.ending = False
        elif isinstance(frame, LLMTextFrame) and frame.text:
            self._buf.append(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame) and self._buf:
            raw = "".join(self._buf)
            self._buf = []
            m = _STEP_RE.match(raw)
            if m:
                self._session.steps_asked.add(int(m.group(1)))
            text = _STEP_ANY_RE.sub("", _END_RE.sub("", raw)).strip()
            if self._session.end_after_reply:
                self._session.ending = True
            elif _END_RE.search(raw):
                if "?" in text[-60:]:
                    logger.info(f"[{self._session.call_id}] END_CALL ignored — response ends with a question")
                else:
                    self._session.ending = True
            elif "?" not in text[-80:] and _GOODBYE_LINE.search(text[-160:]) and self._session.user_turns >= 1:
                logger.info(f"[{self._session.call_id}] closing line without <END_CALL> — ending")
                self._session.ending = True
            if text:
                self._session.bot_texts.append(text)
                self._session.record("bot", text)
        await self.push_frame(frame, direction)


class ChirpTTS(GoogleTTSService):
    """Chirp3-HD that never speaks the <END_CALL> token, skips punctuation-only
    fragments, and applies the "us" respell."""

    def __init__(self, *, session: "CallSession", **kwargs):
        super().__init__(**kwargs)
        self._session = session

    async def run_tts(self, text: str, context_id: str):
        text = self._session.clean_tts_text(text)
        if not text:
            yield None
            return
        async for frame in super().run_tts(_US_FIX.sub("uss", text), context_id):
            yield frame


class CallObserver(BaseObserver):
    """Cost + latency meter, and the hang-up after the final line."""

    def __init__(self, session: "CallSession"):
        super().__init__()
        self._session = session
        self._seen: set[int] = set()
        self.tts_chars = 0
        self.tts_in_tokens = 0     # Gemini TTS: text tokens in
        self.tts_out_tokens = 0    # Gemini TTS: audio tokens out (the billed unit)
        self.tts_provider = "chirp"
        self.llm_in = 0
        self.llm_out = 0
        self.latencies: list[float] = []
        self.tts_ttfbs: list[float] = []   # ElevenLabs: seconds to first audio per line
        self.stt_provider = "sarvam"

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        if isinstance(frame, MetricsFrame):
            if frame.id in self._seen:
                return
            self._seen.add(frame.id)
            for d in frame.data:
                if isinstance(d, TTSUsageMetricsData):
                    self.tts_chars += d.value
                elif isinstance(d, LLMUsageMetricsData):
                    self.llm_in += d.value.prompt_tokens
                    self.llm_out += d.value.completion_tokens
        elif isinstance(frame, BotStoppedSpeakingFrame) and self._session.ending:
            await self._session.hangup("agent finished")

    def cost(self, minutes: float) -> dict:
        if self.tts_provider in RATE_GEMINI_TTS_USD_PER_1M_AUDIO_TOK:
            tts = (self.tts_out_tokens * RATE_GEMINI_TTS_USD_PER_1M_AUDIO_TOK[self.tts_provider]
                   + self.tts_in_tokens * RATE_GEMINI_TTS_USD_PER_1M_TEXT_TOK) / 1e6 * USD_INR
        elif self.tts_provider == "eleven":
            tts = self.tts_chars / 1e6 * RATE_ELEVEN_USD_PER_1M_CHARS * USD_INR
        else:
            tts = self.tts_chars / 1e6 * RATE_CHIRP_USD_PER_1M_CHARS * USD_INR
        stt = minutes / 60 * RATE_SARVAM_STT_INR_PER_HOUR
        llm = (self.llm_in * RATE_LLM_USD_IN_PER_1M + self.llm_out * RATE_LLM_USD_OUT_PER_1M) / 1e6 * USD_INR
        total = tts + stt + llm
        return {
            "total_inr": round(total, 4), "tts_inr": round(tts, 4), "stt_inr": round(stt, 4),
            "llm_inr": round(llm, 4), "minutes": round(minutes, 2),
            "inr_per_min": round(total / minutes, 3) if minutes else None,
            "tts_chars": self.tts_chars, "llm_in": self.llm_in, "llm_out": self.llm_out,
            "tts_provider": self.tts_provider, "tts_out_tokens": self.tts_out_tokens,
            "stt_provider": self.stt_provider,
            "tts_ttfb_max": round(max(self.tts_ttfbs), 2) if self.tts_ttfbs else None,
            "tts_slow_lines": sum(1 for t in self.tts_ttfbs if t >= 1.5),
        }


class CallSession:
    """One call: builds the pipeline for an already-created transport and runs it
    until the far end hangs up, the agent finishes, or the time limit hits."""

    def __init__(self, *, call_id: str, agent: dict, params: dict,
                 transport: FastAPIWebsocketTransport, vad_analyzer: SileroVADAnalyzer,
                 caption_ws=None):
        self.call_id = call_id
        self.agent = agent
        self.params = params
        self.transport = transport
        self.vad_analyzer = vad_analyzer
        self.caption_ws = caption_ws  # browser test page: JSON captions on the same socket
        self.ending = False
        self.user_spoke = False      # set per user turn; FillerPlayer consumes it
        self.filler_until = 0.0      # monotonic time the current filler clip ends
        self.bot_speaking = False    # mirrored by the gate; backchannels wait for silence
        self.backchannel_until = 0.0  # a bare "hmm/achha" transcript before this is echo
        self.last_activity = time.monotonic()  # user words or Genie speech (silence hang-up)
        self.bot_spoke = False       # Genie's first audio has played
        self.awaiting_reply = False  # a user turn was sent and Genie hasn't finished answering
        self.filler_playing = False  # the bot audio playing right now is a hum, not a reply
        self.bot_texts: list[str] = []  # what Genie has said (repeat guard)
        self.user_turns = 0          # transcripts accepted (nudge: once per turn)
        self.nudging = False         # the current bot audio is the "aap sun rahe hain?" nudge
        self.last_vad_at = 0.0       # last VAD start/stop (nudge waits for real quiet)
        self.user_speaking = False
        self.end_after_reply = False # user asked to hang up: end after the next bot line
        self.refund_asks = 0         # "paise wapas?" count — the second one closes the call
        self.openers_used: set[str] = set()  # stock openers ("samajh gayi") already spoken once
        self.steps_asked: set[int] = {1}   # CALL FLOW steps Genie has asked (from the §N markers)
        self.must_ask: dict[int, str] = {int(k): str(v) for k, v in (agent.get("must_ask_steps") or {}).items()}
        self.bot_started_at = 0.0
        self.started = time.monotonic()
        self.task: PipelineTask | None = None
        self.observer = CallObserver(self)
        self.hangup_reason = ""

    # -- TTS helpers (shared by every provider) --
    @staticmethod
    def clean_tts_text(text: str) -> str:
        """Never speak the <END_CALL> sentinel; skip punctuation-only fragments."""
        text = _STEP_ANY_RE.sub("", _END_RE.sub("", text or ""))
        text = _EP_IN_TEXT.sub(r"Episode \1", text)
        return "" if _PUNCT_ONLY.match(text) else text

    def touch(self):
        self.last_activity = time.monotonic()

    def add_tts_usage(self, text_tokens: int, audio_tokens: int):
        self.observer.tts_in_tokens += text_tokens
        self.observer.tts_out_tokens += audio_tokens

    # -- transcript --
    def record(self, role: str, text: str):
        logger.info(f"[{self.call_id}] {role}: {text}")
        store.add_turn(self.call_id, role, text)
        if self.caption_ws is not None:
            asyncio.ensure_future(self._send_caption(role, text))

    async def _send_caption(self, role: str, text: str):
        try:
            await self.caption_ws.send_text(json.dumps({"type": role, "text": text}))
        except Exception:
            pass

    async def hangup(self, reason: str, hard: bool = False):
        """Polite: EndFrame drains queued audio first. Hard: if the call is still up
        3 s later (audio queue stuck / TTS still streaming), cancel the pipeline —
        the call ends no matter what."""
        if self.hangup_reason or not self.task:
            return
        self.hangup_reason = reason
        logger.info(f"[{self.call_id}] hangup: {reason}")
        await self.task.queue_frame(EndFrame())
        if hard:
            await asyncio.sleep(3)
            if self.task and not self.task.has_finished():
                logger.warning(f"[{self.call_id}] still up 3 s after EndFrame — cancelling")
                await self.task.cancel()

    # -- run --
    def _opening_frames(self, greeting: str, pre, lang: str, params: dict, agent: dict,
                        provider: str) -> list:
        """Audio for the scripted opening, fastest first: the per-number pregen; else
        the whole line from the clip cache; else the cached intro sentence followed by
        TTS for the rest; else plain TTS."""
        if pre:
            return [PreRenderedSpeechFrame(pcm=pre[1])]
        if provider not in GEMINI_TTS_MODELS:
            return [TTSSpeakFrame(greeting)]
        _, voice, style = tts_identity(agent, params)
        whole = cached_clip(greeting, provider, voice, style)
        if whole:
            logger.info(f"[{self.call_id}] opening from clip cache")
            return [PreRenderedSpeechFrame(pcm=whole)]
        intro = INTRO.get(lang, {}).get(str(params.get("direction") or "outbound"), "")
        if intro and greeting.startswith(intro):
            clip = cached_clip(intro, provider, voice, style)
            rest = greeting[len(intro):].strip()
            if clip and rest:
                logger.info(f"[{self.call_id}] opening: cached intro + TTS for the rest")
                return [PreRenderedSpeechFrame(pcm=clip), TTSSpeakFrame(rest)]
        return [TTSSpeakFrame(greeting)]

    async def run(self):
        agent, params = self.agent, self.params
        lang = resolve_language(agent, params)
        code = LANG_CODES[lang]
        voice = f"{code}-Chirp3-HD-{params.get('voice') or agent['voice']}"
        creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "gcp-service-account.json"
        system_prompt = render_prompt(agent, params)

        # Telling Sarvam the language matters: on auto-detect it wrote Hindi callers'
        # speech in Bengali/other scripts, which the gate then dropped as noise.
        # STT A/B: param `stt=eleven` (or agent stt.provider / env STT_PROVIDER) runs
        # ElevenLabs Scribe v2 realtime instead of Sarvam. Offline on 6 recordings (12 Oct
        # QC, Gemini as reference) Scribe matched the user's words better on 2 calls,
        # the same on 3, worse on 1 — not a clear win, hence a switch, not a change.
        stt_provider = str(params.get("stt") or (agent.get("stt") or {}).get("provider")
                           or os.getenv("STT_PROVIDER", "sarvam")).lower()
        if stt_provider == "eleven" and os.getenv("ELEVENLABS_API_KEY"):
            stt = ElevenLabsRealtimeSTTService(
                api_key=os.getenv("ELEVENLABS_API_KEY"),
                settings=ElevenLabsRealtimeSTTService.Settings(model=os.getenv("ELEVENLABS_STT_MODEL", "scribe_v2_realtime"),
                                                               language=Language(code)),
                sample_rate=INPUT_SAMPLE_RATE,
            )
        else:
            stt_provider = "sarvam"
            stt = SarvamSTTService(
                api_key=os.getenv("SARVAM_API_KEY"),
                settings=SarvamSTTService.Settings(model=STT_MODEL, language=Language(code)), mode="codemix",
                sample_rate=INPUT_SAMPLE_RATE,
            )
        self.observer.stt_provider = stt_provider
        llm = GoogleLLMService(
            api_key=os.getenv("GOOGLE_API_KEY"),
            settings=GoogleLLMSettings(
                model=LLM_MODEL, system_instruction=system_prompt,
                temperature=float(params.get("temperature") or agent["temperature"]),
                max_tokens=300,
            ),
        )
        # speaking_rate is the Chirp3-HD speed knob (0.25–2.0; SSML prosody isn't
        # supported by Chirp). Per-call `speed` param wins over the agent default.
        speed = max(0.25, min(2.0, float(params.get("speed") or agent["speed"] or 1.0)))
        tts_cfg = agent.get("tts") or {}
        provider = str(params.get("tts") or tts_cfg.get("provider") or "chirp").strip().lower()
        if provider not in TTS_PROVIDERS:
            provider = "chirp"
        eleven_voice = str(params.get("voice_id") or tts_cfg.get("voice_id") or os.getenv("ELEVENLABS_VOICE_ID") or "")
        if provider == "eleven" and not (eleven_voice and os.getenv("ELEVENLABS_API_KEY")):
            logger.warning(f"[{self.call_id}] tts=eleven but no ELEVENLABS_API_KEY / voice id — falling back to chirp")
            provider = "chirp"
        self.observer.tts_provider = provider
        http = aiohttp.ClientSession() if provider in GEMINI_TTS_MODELS else None
        if provider == "chirp":
            tts = ChirpTTS(
                session=self, credentials_path=creds, voice_id=voice,
                params=GoogleTTSService.InputParams(language=Language(code), speaking_rate=speed),
                sample_rate=OUTPUT_SAMPLE_RATE,
            )
        elif provider == "eleven":
            # Eleven v4 Turbo over the text-to-dialogue websocket (see eleven_tts.py);
            # voice is an ElevenLabs voice id, language hint from the call language.
            tts = ElevenTTS(
                session=self, api_key=os.getenv("ELEVENLABS_API_KEY", ""), voice_id=eleven_voice,
                model=str(params.get("tts_model") or tts_cfg.get("model") or ELEVEN_DEFAULT_MODEL),
                language=code.split("-")[0], sample_rate=OUTPUT_SAMPLE_RATE,
            )
        else:
            # Gemini picks the language from the (native-script) text; voice is the bare
            # name (Sulafat), pace goes in the style prompt — there is no rate field.
            tts = GeminiTTS(
                session=self, api_key=os.getenv("GOOGLE_API_KEY"), aiohttp_session=http,
                model=GEMINI_TTS_MODELS[provider], voice=params.get("voice") or agent["voice"],
                style=str(params.get("tts_style") or tts_cfg.get("style") or ""),
                sample_rate=OUTPUT_SAMPLE_RATE,
            )
        shown_voice = voice if provider == "chirp" else eleven_voice if provider == "eleven" else (params.get("voice") or agent["voice"])
        logger.info(f"[{self.call_id}] agent={agent['id']} lang={lang} tts={provider} voice={shown_voice} "
                    f"speed={speed} llm={LLM_MODEL} stt={stt_provider}")

        # How long the user must be quiet before their turn ends. 0.6 s made Genie jump
        # in mid-sentence ("मुझे तो…", "ठीक ही लग रहा था मतलब…"); people pause to think.
        # When the transcript so far looks unfinished (ends on "matlab", "aur", "ki"…)
        # the strategy waits turn_hold_secs instead — see UnfinishedAwareTurnStop.
        turn_end_secs = float(params.get("turn_end_secs") or agent.get("turn_end_secs") or 1.2)
        turn_hold_secs = float(params.get("turn_hold_secs") or agent.get("turn_hold_secs") or 2.5)

        # Genie speaks first: seed one hidden user turn so the model opens the call.
        kickoff = f"(The call connected. Greet the user in {LANG_NAMES[lang]}.)"
        context = LLMContext(messages=[{"role": "user", "content": kickoff}])
        aggregators = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    start=[VADUserTurnStartStrategy(enable_interruptions=False),
                           TranscriptionUserTurnStartStrategy(use_interim=False,
                                                              enable_interruptions=False)],
                    stop=[UnfinishedAwareTurnStop(base=turn_end_secs, hold=turn_hold_secs)],
                ),
                user_turn_stop_timeout=2.5,
            ),
        )

        # Turn-end fillers and backchannel are independent switches; both play through
        # one FillerPlayer (after the user aggregator), the backchannel is driven by a
        # SpeechWatcher right after the VAD. Per-call params `fillers` / `backchannel`.
        filler_cfg = agent.get("fillers") or {}
        bc_cfg = dict(agent.get("backchannel") or {})
        turn_fillers = bool(filler_cfg.get("enabled")) and str(params.get("fillers", "1")) != "0"
        if "backchannel" in params:
            bc_cfg["enabled"] = str(params["backchannel"]) not in ("0", "false", "off")
        player, trigger, watcher = [], [], []
        if turn_fillers or bc_cfg.get("enabled"):
            fp = FillerPlayer(self, lang, turn_fillers=turn_fillers,
                              every=filler_cfg.get("every", 2), backchannel=bc_cfg,
                              hums_only=bool(filler_cfg.get("hums_only")),
                              filler_gain=float(filler_cfg.get("gain", 0.8)),
                              delay_secs=float(filler_cfg.get("delay_secs", 0)))
            player = [fp]
            if turn_fillers:
                trigger = [TurnEndTrigger(fp, self)]
            if bc_cfg.get("enabled"):
                watcher = [SpeechWatcher(fp, self, bc_cfg)]

        # Barge-in: word-gated in TranscriptionGate (VAD alone would let echo and
        # background noise cut Genie off). Per-call param `interrupt=0|1`.
        int_cfg = agent.get("interruptions") or {}
        barge_in = bool(int_cfg.get("enabled"))
        if "interrupt" in params:
            barge_in = str(params["interrupt"]) not in ("0", "false", "off")
        interrupt_words = int(int_cfg.get("min_words", 2)) if barge_in else 0

        pipeline = Pipeline([
            self.transport.input(),
            VADProcessor(vad_analyzer=self.vad_analyzer),
            *watcher,
            stt,
            TranscriptionGate(self, interrupt_words, lang),
            UserTurnRecorder(self),
            aggregators.user(),
            *trigger,
            llm,
            ReplyGuard(self, lang),
            BotTurnRecorder(self),
            tts,
            *player,                  # second channel: fillers + backchannel
            self.transport.output(),
            aggregators.assistant(),
        ])

        latency = UserBotLatencyObserver()

        @latency.event_handler("on_latency_measured")
        async def _on_latency(observer, secs, *args):
            self.observer.latencies.append(secs)
            logger.info(f"[{self.call_id}] latency {secs:.2f}s")

        self.task = PipelineTask(
            pipeline,
            params=PipelineParams(allow_interruptions=barge_in, enable_metrics=True,
                                  enable_usage_metrics=True),
            observers=[self.observer, latency],
            idle_timeout_secs=90,
        )

        pre = take_pregen(self.params.get("_number", "")) if player else None
        greeting = pre[0] if pre else scripted_greeting(lang, params)

        @self.transport.event_handler("on_client_connected")
        async def _connected(transport, client):
            if not agent["greeting_first"]:
                return
            if greeting:
                # People pick up and take a beat to bring the phone to the ear; starting
                # the instant Exotel connects ("aap turant start kar jaati hain") loses
                # the first words. Inbound callers are already listening.
                # Most people also say "hello?" as they pick up. If the greeting is
                # already playing, that hello is ignored and they say it again — twice,
                # three times — and miss the opening (12 Oct QC). So: wait for their
                # hello to finish (up to greeting_wait_secs), then greet.
                if str(params.get("direction", "")).lower() != "inbound":
                    delay = float(params.get("greeting_delay_secs") or agent.get("greeting_delay_secs") or 1.0)
                    wait = float(params.get("greeting_wait_secs") or agent.get("greeting_wait_secs") or 2.5)
                    t0 = time.monotonic()
                    heard = False
                    while time.monotonic() - t0 < wait:
                        if self.last_vad_at >= t0:
                            heard = True
                            if not self.user_speaking and time.monotonic() - self.last_vad_at >= 0.3:
                                break
                        await asyncio.sleep(0.1)
                    if not heard:
                        await asyncio.sleep(max(0.0, delay - (time.monotonic() - t0)))
                    logger.info(f"[{self.call_id}] greeting after {time.monotonic() - t0:.1f}s "
                                f"({'they spoke first' if heard else 'silence'})")
                self.bot_texts.append(greeting)
                self.record("bot", greeting)
                await self.task.queue_frames([
                    LLMMessagesAppendFrame(messages=[{"role": "assistant", "content": greeting}],
                                           run_llm=False)])
                # Pre-rendered audio rides the task queue (after StartFrame, in order);
                # pushing it straight into the output at connect time got lost.
                await self.task.queue_frames(self._opening_frames(greeting, pre, lang, params, agent,
                                                                  provider))
            else:
                await self.task.queue_frames([LLMRunFrame()])

        @self.transport.event_handler("on_client_disconnected")
        async def _disconnected(transport, client):
            self.hangup_reason = self.hangup_reason or "far end hung up"
            await self.task.cancel()

        max_secs = min(int(params.get("max_call_secs") or agent["max_call_secs"]), 300)  # 5 min, always

        async def _time_limit():
            await asyncio.sleep(max_secs)
            await self.hangup(f"time limit {max_secs}s", hard=True)

        # Hang up when nobody has said anything for `silence_hangup_secs` (default 60):
        # the user went quiet and Genie isn't talking either.
        silence_secs = float(params.get("silence_hangup_secs") or agent.get("silence_hangup_secs") or 60)

        nudge_text = NUDGE.get(lang)
        nudge_after = float(params.get("nudge_after_secs") or agent.get("nudge_after_secs") or 7)

        wrap_up_secs = float(params.get("wrap_up_secs") or agent.get("wrap_up_secs") or 150)

        async def _wrap_up():
            # Calls were running 4-5 minutes; at 2.5 min steer the model to close.
            await asyncio.sleep(wrap_up_secs)
            if not self.hangup_reason and not self.ending:
                logger.info(f"[{self.call_id}] {wrap_up_secs:.0f}s — wrap-up note")
                await self.task.queue_frame(LLMMessagesAppendFrame(messages=[{
                    "role": "user",
                    "content": "(Time is up. From now on: at most ONE more short question — the closing "
                               "one — then thank them and end with <END_CALL>. If they are mid-story, "
                               "let them finish, acknowledge, and close.)"}], run_llm=False))

        async def _silence_watch():
            nudged_turn = -1
            while not self.hangup_reason:
                await asyncio.sleep(1)
                now = time.monotonic()
                # Nobody said anything for `nudge_after` s after Genie's question: check
                # they're still there — once per user turn; it doesn't reset the silence clock.
                # …and never while Genie still owes a reply: with slow TTS the nudge used
                # to land right before the answer ("aap sun rahe hain?" + reply = two replies).
                if (nudge_text and not self.bot_speaking and not self.ending and self.bot_spoke
                        and not self.user_speaking and not self.awaiting_reply
                        and now - self.last_vad_at >= nudge_after
                        and now - self.last_activity >= nudge_after and nudged_turn != self.user_turns):
                    nudged_turn = self.user_turns
                    self.nudging = True
                    logger.info(f"[{self.call_id}] silence {nudge_after:.0f}s — nudge")
                    self.record("bot", nudge_text)
                    await self.task.queue_frame(TTSSpeakFrame(nudge_text))
                    continue
                if self.bot_speaking and now - self.bot_started_at > 60:
                    # No reply is a minute long — the TTS stream is stuck / babbling.
                    await self.hangup("bot audio stuck >60s", hard=True)
                    return
                if not self.bot_speaking and now - self.last_activity >= silence_secs:
                    await self.hangup(f"user silent {silence_secs:.0f}s")
                    return

        limit = asyncio.gather(_time_limit(), _silence_watch(), _wrap_up())
        limit.add_done_callback(lambda f: f.cancelled() or f.exception())  # swallow the cancel noise
        status, error = "ended", ""
        try:
            await PipelineRunner(handle_sigint=False).run(self.task)
        except Exception as e:  # noqa: BLE001 — one call's failure must not take the server down
            status, error = "failed", str(e)
            logger.exception(f"[{self.call_id}] pipeline error: {e}")
        finally:
            limit.cancel()
            if http is not None:
                await http.close()
            minutes = max((time.monotonic() - self.started) / 60, 1 / 60)
            cost = self.observer.cost(minutes)
            lat = sorted(self.observer.latencies)
            p50 = lat[len(lat) // 2] if lat else None
            store.end_call(self.call_id, status=status, cost=cost, latency_p50=p50,
                           error=error or self.hangup_reason)
            logger.info(f"[{self.call_id}] {status} · ₹{cost['total_inr']:.2f} "
                        f"({cost['inr_per_min']} ₹/min) · p50 {p50}")


def make_vad() -> SileroVADAnalyzer:
    # stop_secs 0.4 → 0.7: a breath mid-sentence no longer ends the user's turn.
    return SileroVADAnalyzer(params=VADParams(confidence=0.6, start_secs=0.2,
                                              stop_secs=0.5, min_volume=0.3))
