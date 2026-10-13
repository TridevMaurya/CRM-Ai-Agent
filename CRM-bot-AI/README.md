# CRM bot AI — Genie, the voice research agent

An AI caller ("Genie") that phones app users, has a short natural Hindi conversation
about a video they watched (or why they cancelled), and turns every call into
structured research data on a dashboard. Built for Gyan TV; works for any product
that has a user sheet and a content sheet.

```
CRM-bot-AI/
├── dashboard/        the web UI: live calls, transcripts, recordings, insights, filters,
│                     Excel export (index.html), per-call page (call.html), admin panel
│                     for prompts/voice/timing (admin.html), browser test call (test.html)
├── calling-agent/    what Genie says and how she behaves
│   ├── agents/       one YAML per agent: prompt, call flow, cohort/theme probes, analysis
│   │                 schema, voice, timing  (genie-content-feedback, genie-cancellation)
│   └── fillers/      the recorded "hmm" sounds she plays while thinking
├── call-server/      the machine that connects and runs the calls
│   ├── app/          FastAPI server: Exotel/browser audio in → Silero VAD → Sarvam STT →
│   │                 Gemini LLM → TTS (ElevenLabs / Chirp / Gemini) → audio out;
│   │                 campaign dialer, Google Sheets sync, SQLite, post-call analysis
│   ├── scripts/      deploy, QC of recordings, research coding, e2e browser call
│   └── sample_sheets/ example user + content sheets (CSV) to start from
├── Dockerfile, fly.toml, requirements.txt
└── .env.example      every setting, with comments
```

## Run it yourself

1. **Python 3.13**, then from this folder:
   ```bash
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env            # fill in the keys below
   ```
2. **Keys** (in `.env`):
   - `GOOGLE_API_KEY` — Gemini (the brain + post-call analysis + video summaries)
   - `SARVAM_API_KEY` — Sarvam saaras speech-to-text (Hindi/Indic)
   - `ELEVENLABS_API_KEY` + `ELEVENLABS_VOICE_ID` — the voice (Eleven v4 Turbo).
     Or use Google Chirp3-HD: put a service-account JSON at
     `gcp-service-account.json` (Cloud Text-to-Speech + Sheets scopes) and set
     `tts.provider: chirp` in the agent YAML / admin panel.
   - `EXOTEL_API_KEY`, `EXOTEL_API_TOKEN`, `EXOTEL_ACCOUNT_SID` — phone calls (outbound
     dialing + inbound). Without Exotel you can still test in the browser at `/test/<agent>`.
   - `ADMIN_PASSWORD` — for `/admin`; `DASHBOARD_TOKEN` — optional lock on the dashboard.
3. **Sheets** (`USER_SHEET`, `CONTENT_SHEET` in `.env`): share both with the service
   account email shown on the dashboard. Column names are documented in
   `.env.example`; `call-server/sample_sheets/` shows the shape. Scripts in the
   per-language content tabs become the 10-second video summaries Genie uses to ask
   specific questions.
4. **Start**:
   ```bash
   cd call-server && python -m app.main        # http://127.0.0.1:8080
   ```
   Dashboard at `/`, admin at `/admin`, health at `/health`. The dialer is off until
   you switch it on from the dashboard (or `POST /api/campaign/dialer {"enabled": true}`).
5. **Deploy to Fly.io**: `call-server/scripts/deploy_fly.sh` (reads `.env` and the
   service-account file, pushes them as secrets, builds the Dockerfile).

## How a call works

1. The dialer picks the next contact (cohort order, theme priority, optional focus by
   last-video category), pre-renders the opening line, and dials through Exotel.
2. Audio streams over a WebSocket into the pipeline; Genie speaks first, within ~0.5 s.
3. Each reply follows the agent's CALL FLOW (video in detail → other apps → why this
   app → how the app runs → close). A hidden step marker on every reply lets the
   server remind the model of steps it hasn't asked yet.
4. After the call, Gemini fills the agent's analysis schema (sentiment, reasons,
   wanted video, proof needed, other-app gaps, …); the dashboard buckets it and the
   call is appended to the user sheet's `call_log` tab.

## Editing Genie

- Prompts, probes per cohort/theme, voice, pace and timing: `calling-agent/agents/*.yaml`,
  or live from `/admin` (saved as overrides, YAML stays the default).
- The analysis schema at the bottom of each YAML drives the dashboard columns and export.

No real user data is in this repo: the SQLite database, recordings, `.env` and the
service-account file are git-ignored.
