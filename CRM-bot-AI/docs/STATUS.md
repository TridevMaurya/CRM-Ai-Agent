# Status / work log

Owner: Tridev · Ask: AI "Genie" calls Gyan TV viewers a few hours after they watch a
video and collects feedback (Saloni / Akhila / Naveen thread); v2 = retargeting and
cancellation-reason calls.

## Done (22 Sep 2026)

**Platform**
- [x] Multi-agent voice server (FastAPI + Pipecat 1.1): Sarvam saaras STT → Gemini 3.1
      flash-lite → Google Chirp3-HD TTS. One WebSocket URL per agent, ElevenLabs-style.
- [x] Agents as YAML (`agents/`): prompt, voice, speed, language, defaults, analysis fields.
      Two agents: content feedback, cancellation reason.
- [x] Dashboard: agents + WS URLs, Dial / Save-context form, recent calls with status,
      duration, cost, latency, sentiment, summary; call page with transcript + analysis.
- [x] Browser test page (mic) — same pipeline as the phone.
- [x] Per-call cost meter (TTS chars, STT time, Gemini tokens → ₹) and latency p50.
- [x] Post-call analysis → JSON (liked / met expectations / complaints / wants more of /
      sentiment / summary / do-not-call), shown on the dashboard.
- [x] Call ends itself (`<END_CALL>`), hard limit 4 min, 90 s idle timeout.
- [x] Guards: foreign-script STT hallucinations dropped; no barge-in (echo safe); END_CALL
      ignored if the response is a question; native-script rule (Gemini wrote Telugu in
      Roman letters without it).
- [x] Context precedence: YAML defaults → dashboard default → number-specific → Exotel
      custom params. Inbound vs outbound detected; inbound gets a "thanks for calling" note.
- [x] Automated end-to-end call test (`scripts/e2e_browser_call.py`), Hindi + Telugu pass.

**Deploy**
- [x] Fly.io app `gyan-tv-crm-glapp8` (sin), volume `crm_data`, secrets pushed
      (Gemini, Sarvam, GCP service account, Exotel). `scripts/deploy_fly.sh`.

**Exotel**
- [x] ExoPhone 08047285306 → custom app "app peer to teacher" (1312681) → Voicebot applet →
      Genie content-feedback URL, recording ON. Incoming works.
- [x] Outbound Dial from the dashboard through the same flow (Connect API).
- [x] Default context for incoming calls: Naveen / "Bangalore horror story" / Hindi.

**Voice settings**: Chirp3-HD **Sulafat**, speed **0.9** (both agents).

## Real phone calls so far (22 Sep, all incoming from one test number)

| Time | Length | Result |
|---|---|---|
| 02:39 | 116 s | Full conversation, Genie ended the call. Feedback: weak story, not scary, wants fashion content. Analysis correct. ₹3.41 (₹1.78/min), latency p50 2.2 s |
| 02:47 | 10 s | Caller hung up after the greeting |
| 02:51 | 105 s | Liked graphics/audio, video too long. Caller hung up on the last question. ₹3.74 |
| 02:54 | 40 s | Caller asked "which app is calling"; Genie answered "Gyan TV's official number", caller hung up |

No pipeline errors. Phone latency matches the browser tests (~2.1–2.2 s).

## Open issues (small)

1. **Gender forms** — Genie sometimes uses masculine Hindi verbs ("समझ गया", "पहुँचाऊँगा").
   Fix: prompt rule "you are female — feminine forms only".
2. **Double replies** — a pause mid-sentence splits one user turn into two, Genie answers
   twice (seen 02:51). Fix: VAD `stop_secs` 0.4 → ~0.7 for phone (+0.3 s latency).
3. **"Who is calling?"** — Genie should say "AI assistant Genie on behalf of Gyan TV"; the
   prompt says so but she answered only "Gyan TV's official number". Tighten the rule.
4. Inbound greeting is neutral rather than "thanks for calling back" — acceptable, tune later.
5. Sarvam STT occasionally emits a stray Tamil/other-script fragment ("ஆ"); harmless, the
   LLM ignores it.

## Next

- Fix the three prompt/VAD items above, re-test with 2–3 team calls in Hindi and Telugu.
- Team listens to a few recordings (Exotel) and picks the voice (Sulafat vs Aoede/Achird).
- Dialer for real usage: pick viewers who watched a video N hours ago → `POST /api/dial`
  with `user_name`, `video_title`, `language` (needs the watch-event feed from the app).
- Push the repo to GitHub (no remote yet); optional `DASHBOARD_TOKEN` before sharing widely.
- Cost check: compare the meter's ₹ against Sarvam / Google / Gemini billing after ~100 calls.
- v2: cancellation-reason calls (agent exists; needs the cancellation event + its own
  Exotel flow if it should also take incoming), retargeting.
