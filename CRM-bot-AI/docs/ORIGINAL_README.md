# Gyan TV CRM — Genie voice agents

AI calling agents for Gyan TV, built like ElevenLabs Conversational AI but on our own
cheap stack: **every agent gets a WebSocket URL** you paste into an Exotel Voicebot
applet, plus a dashboard with transcripts, per-call cost and a structured analysis of
what the user said.

**Live:** https://gyan-tv-crm-glapp8.fly.dev · **Phone:** 080-4728-5306 (Exotel) ·
**Status / work log:** [STATUS.md](STATUS.md)

```
Exotel (phone) ──WS──▶  /ws/<agent>  ──▶  Sarvam saaras STT ──▶ Gemini Flash-Lite ──▶ Google Chirp3-HD TTS
Browser test   ──WS──▶  /ws/browser/<agent>        (same pipeline, only the transport differs)
                                       │
                                       ▼
                         SQLite: calls · transcript turns · post-call analysis (JSON)
                                       │
                                       ▼
                              Dashboard  /   ·   /calls/<id>
```

Agents live in `agents/*.yaml` — prompt, voice, speed, language, default context, and
the fields the post-call analysis extracts. Two ship today:

| Agent | What it does |
|---|---|
| `genie-content-feedback` | A few hours after a user watched a video: did it meet expectations, what was missing, what do they want more of |
| `genie-cancellation` | After a cancellation: the real reason, the trigger, what would bring them back |

Both currently speak Hindi by default (Telugu/Tamil/Kannada/English via the `language`
param), voice **Chirp3-HD Sulafat**, speed **0.9**.

## Run locally

```bash
python3.13 -m venv venv && venv/bin/pip install -r requirements.txt
cp .env.example .env            # fill GOOGLE_API_KEY, SARVAM_API_KEY (+ EXOTEL_* for dialing)
cp <your>/gcp-service-account.json .   # Google Cloud TTS service account
venv/bin/python -m app.main     # http://127.0.0.1:8080
```

Open the dashboard → **Test in browser** → talk to Genie. The call appears in *Recent
calls* with transcript, cost and analysis a few seconds after it ends.

Automated end-to-end call (no mic needed — the "user" is Sarvam-rendered audio):

```bash
venv/bin/python scripts/e2e_browser_call.py http://127.0.0.1:8080 genie-content-feedback hin
```

## How a call gets its context

Genie needs to know who she is talking to and about which video. Sources, lowest to
highest precedence:

1. `defaults:` in the agent YAML (today: Naveen / "Bangalore horror story" / Hindi).
2. Default context saved on the dashboard (**Save context only** with the number blank) —
   overrides the YAML without a redeploy.
3. Number-specific context: `POST /api/context {"agent_id","number","params"}` or the
   dashboard **Dial** form. Picked up when that number connects (valid 2 h, used once).
4. `custom_parameters` set on the Exotel Voicebot applet.

A call with number-specific context is treated as **outbound** (we dialed); one without
is **inbound** (the user called the ExoPhone) and Genie's prompt gets a note to thank
them for calling instead of saying she is calling them. An inbound call from a number in
the campaign queue gets that contact's agent and context (they're returning our call).

## Campaign: sheets → auto-dialer (`app/campaign.py`)

```
content sheet ─┐                                  ┌─> Exotel dials → Genie (agent by cohort)
               ├─ sync every 15 min → contacts ─ dialer
user sheet ────┘   (SQLite queue)                  └─> result appended to the call_log tab
```

- **Content sheet** (`CONTENT_SHEET`): header row with `Content ID`, `Video-Title`,
  `Category` (+ optional `Script` or `Summary`; a script gets a ~10 s spoken summary
  generated once). Title rows above the header are fine.
- **User sheet** (`USER_SHEET`, tab `user_queue`, refreshed hourly from BigQuery by the
  analyst): `phone`, `name`, `language`, `cohort`, `watched_content_ids`,
  `last_watched_content_id`, `top_category`, `last_content_before_cancel`, `plan_price`, ….
- **Cohort → agent** via `cohorts:` in each agent YAML: `LIGHT_3_4`, `DAU_10PLUS` → content
  feedback; `TRIAL_CANCELLED`, `SUB_CANCELLED` → cancellation. `cohort_prompts:` holds each
  cohort's research questions.
- **Dialer**: off until **Start dialer** on the dashboard. Only inside `CALL_WINDOW` (IST),
  ≤ `MAX_CONCURRENT_CALLS`, retries no-answers (`MAX_ATTEMPTS`, `RETRY_AFTER_MINS`), never
  re-calls a number within `RECALL_AFTER_DAYS`, never again after "don't call me".
- Share both sheets with the service account (Editor on the user sheet for `call_log`),
  and enable the **Google Sheets API** in the GCP project.

## Admin panel (`/admin`)

Log in with `ADMIN_PASSWORD` (Fly secret, pushed by `scripts/deploy_fly.sh`). Per agent:
TTS engine, voice, style, speed, language, temperature, max call length, silence
hang-up, barge-in, turn-end fillers, backchannel timing / volume, the main prompt and
each cohort prompt, plus a voice preview. Saves go to SQLite as overrides over the
YAML and apply from the next call — no deploy. "Reset" returns to the YAML.

## TTS engines (`tts:` in the agent YAML)

`chirp` (default) — Google Chirp3-HD, streams per sentence, ~0.3 s to first audio.
`gemini_lite` / `gemini_flash` — Gemini 3.8 TTS over the Interactions API
(`app/gemini_tts.py`): one request per reply, ~1.4 / 1.9 s to first audio, billed per
audio token (~₹1.2 / ₹1.8 per minute of speech). Voice names are shared; `style` is a
Gemini-only delivery prompt. Per call: `tts=gemini_lite`.

## Fillers and backchannel (`app/fillers.py`)

- **Turn-end filler** (`fillers:`): on every 2nd user turn a cached "okay / achha /
  haan ji" plays the instant the user stops talking, covering the LLM + TTS gap.
- **Backchannel** (`backchannel:`): while the user talks, a soft sound every 4–6 s of
  speech (jittered) or at a breath after ≥ 2.4 s — 70 % recorded hums
  (`fillers/hums/`), 30 % word clips, at half volume, max 3 per turn, never while
  Genie speaks. A bare "hmm/achha" transcript right after one is dropped as echo.

Clips are made **once** (`fillers/<lang>/*.wav`, `fillers/hums/*.wav`) and shipped in
the image — no TTS call at runtime:

```bash
venv/bin/python scripts/make_fillers.py --provider chirp --voice Sulafat      # same voice as Genie
venv/bin/python scripts/make_fillers.py --provider elevenlabs --voice <id> --force  # ElevenLabs v3
```

Per call: `fillers=0`, `backchannel=0`.

## API

| Method | Path | |
|---|---|---|
| GET | `/api/agents` | agents, their WS URLs, defaults, call stats |
| GET | `/api/calls?agent_id=&limit=` | recent calls (status, duration, cost, analysis) |
| GET | `/api/calls/{id}` | one call with transcript turns |
| POST | `/api/context` | with `number`: context for that number's next call; without: the agent's default context |
| POST | `/api/dial` | dial via Exotel (stores context too) |
| GET | `/api/campaign` | sheet sync + dialer status, queue counts |
| POST | `/api/campaign/sync` | sync both sheets now |
| POST | `/api/campaign/dialer` | `{"enabled": true/false}` |
| GET | `/api/contacts?status=` | campaign queue |
| POST | `/api/contacts/{id}/dial` | dial one contact now |
| GET | `/health` | |

## Costs and latency (measured on real phone calls, 22 Sep 2026)

**₹1.8–2.2 per talk-minute**, voice-to-voice **~2.1–2.2 s**. Breakdown at ~550 chars/min:
Chirp3-HD ≈ ₹1.45, Sarvam STT ≤ ₹0.50 (priced on call length — upper bound), Gemini ≈
₹0.40. Every call's actual cost is on its detail page; rates are env-overridable
(`RATE_*`, `USD_INR` in `app/pipeline.py`). Phone audio is 8 kHz (Exotel/telephony); we
synthesise at 24 kHz and downsample, the browser test plays the full 24 kHz.

## Adding an agent

Copy an existing YAML in `agents/`, change `id`, `name`, `prompt`, `defaults` and the
`analysis` fields. YAML edits need a deploy (the files ship in the image).
`{{placeholders}}` in the prompt come from the per-call params; `{{x_or_some_words}}`
falls back to "some words" when `x` isn't given. A native-script rule (Devanagari /
Telugu / …) and, for inbound calls, a greeting note are appended automatically.

## Layout

```
app/main.py       FastAPI: pages, API, the two WebSocket endpoints, context precedence
app/pipeline.py   the Pipecat pipeline for one call (STT → LLM → TTS), gate, cost meter, hang-up
app/agents.py     YAML loading + prompt templating (+ script rule, inbound note)
app/store.py      SQLite: calls, turns, agent defaults, pending context
app/analysis.py   post-call JSON extraction (Gemini)
app/exotel.py     outbound dial (Exotel Connect API)
app/campaign.py   sheet sync, contacts queue, auto-dialer, call_log write-back
app/sheets.py     Google Sheets API (service account)
app/fillers.py    cached filler clips played on alternate user turns
agents/           one YAML per agent
fillers/          filler clips per language (scripts/make_fillers.py)
static/           dashboard, call page, browser test page
scripts/          deploy_fly.sh, e2e_browser_call.py, make_fillers.py
```
