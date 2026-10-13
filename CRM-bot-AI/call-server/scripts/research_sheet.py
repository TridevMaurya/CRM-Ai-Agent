"""Write the research workbook to a Google Sheet: every call, every coded
conversation, and summary tabs whose numbers are formulas over those rows.

Usage: python scripts/research_sheet.py <sheet_id> <calls.pkl> <api_calls.json> <coded.jsonl> <findings.json>
  calls.pkl      rows of /api/calls/export.xlsx (see research_code.py)
  api_calls.json /api/calls pages (for call ids → dashboard links)
  coded.jsonl    output of research_code.py
  findings.json  [{"finding": ..., "metric": ..., "formula": ..., "format": "percent"|"integer", "reading": ...}]
Needs the service account to have edit access to the sheet."""
import json
import pickle
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import sheets  # noqa: E402

BASE = "https://gyan-tv-crm-glapp8.fly.dev"
IST = timezone(timedelta(hours=5, minutes=30))
API = "https://sheets.googleapis.com/v4/spreadsheets"

REASONS = {
    "unaware_auto_debit": "Did not know they would be charged",
    "wanted_refund": "Wants a refund",
    "expected_to_earn_money": "Expected to earn money from the app",
    "content_not_useful_or_practical": "Content not useful or practical",
    "content_not_found": "Could not find the content they wanted",
    "content_quality": "Video quality (audio, visuals, length)",
    "misleading_promise": "Content did not match what was promised",
    "language_barrier": "Language was a barrier",
    "app_hard_to_use": "App hard to use",
    "price_too_high": "Price too high",
    "no_money_right_now": "No money right now",
    "no_time_or_not_using": "No time or not using it",
    "technical_issue": "Technical problem",
    "uses_free_alternative": "Uses a free alternative",
    "only_wanted_to_try": "Only wanted to try it",
    "someone_else_subscribed": "Someone else subscribed on their phone",
    "did_not_cancel_or_still_using": "Says they did not cancel or still use it",
    "other": "Other reason",
    "not_stated": "No reason stated",
}
DISLIKED = {
    "not_practical_no_steps": "Not practical, no step-by-step",
    "too_basic_or_generic": "Too basic or generic",
    "misleading_promise": "Promise did not match the content",
    "too_long": "Too long",
    "too_short": "Too short",
    "poor_audio": "Poor audio",
    "poor_visuals": "Poor visuals",
    "hard_language": "Language hard to follow",
    "repetitive": "Repetitive",
    "not_sequenced": "Videos not in sequence",
    "hard_to_find_content": "Hard to find content",
    "paywall_or_ads": "Paywall or ads",
}
REQUESTS = {
    "practical_earning_methods": "Practical ways to earn",
    "govt_exam_prep": "Government exam prep (UPSC, PSC, SSC)",
    "english_speaking": "English speaking",
    "share_market_finance": "Share market and finance",
    "business_ideas": "Business ideas",
    "ai_tech_skills": "AI and tech skills",
    "devotional": "Devotional",
    "entertainment": "Entertainment (songs, serials, films)",
    "health": "Health",
    "agriculture": "Agriculture",
    "regional_language": "Content in their own language",
    "school_education": "School education",
    "history_gk": "History and general knowledge",
    "other": "Other topics",
}
# Codes the model used outside the codebook, folded into the nearest reason.
REASON_ALIASES = {"content_not_sequenced": "content_quality", "not_sequenced": "content_quality",
                  "content_not_practical_no_steps": "content_not_useful_or_practical"}
ENGAGEMENT = {"full": "Answered several questions", "partial": "One real answer",
              "refused_or_busy": "Refused or busy", "confused": "Confused about the call",
              "none": "No engagement"}
RESPONDENT = {"user": "The user", "family_member": "Family member", "wrong_person": "Wrong person",
              "unclear": "Unclear"}
YESNO = {"yes": "Yes", "no": "No", "unclear": "Unclear", "maybe": "Maybe", "not_asked": "Not asked"}
AWARE = {"yes": "Yes", "no_didnt_know_charged": "No, did not know they were charged", "unclear": "Unclear"}
SENTIMENT = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative", "angry": "Angry"}
COHORT = {"TRIAL_CANCELLED": "Trial cancelled", "SUB_CANCELLED": "Subscription cancelled",
          "DAU_10PLUS": "Active viewer (10+ videos)", "LIGHT_3_4": "Light viewer (4-9 videos)"}

CONV_HEADERS = ["Call time (IST)", "Cohort", "Theme they watch", "Last video", "Duration (s)", "Respondent",
                "Engagement", "Knows Gyan TV", "Aware of payment", "Main reason", "All reasons",
                "Amount mentioned", "Refund requested", "Watched content", "Liked", "Disliked",
                "Disliked tags", "Content requests", "Request categories", "Would return", "Price willing",
                "Sentiment", "Quote (their words)", "Quote (English)", "One-line summary",
                "Insight score (0-3)", "Could not hear the caller", "Call link", "Transcript"]
CALL_HEADERS = ["Call time (IST)", "Cohort", "Theme they watch", "Agent", "Status", "How it ended",
                "Duration (s)", "User turns", "User words", "Outcome", "Call link"]


def col(n: int) -> str:
    """1-based column number → letter(s)."""
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


C = {h: col(i + 1) for i, h in enumerate(CONV_HEADERS)}      # Conversations column letters
A = {h: col(i + 1) for i, h in enumerate(CALL_HEADERS)}      # All calls column letters


def cv(h: str) -> str:
    return f"Conversations!${C[h]}$2:${C[h]}"


def ac(h: str) -> str:
    return f"'All calls'!${A[h]}$2:${A[h]}"


def pick(mapping: dict, v, default: str) -> str:
    return mapping.get(str(v or "").strip().lower(), default)


def join(mapping: dict, vs, drop: tuple = ()) -> str:
    out = []
    for v in (vs if isinstance(vs, list) else [vs]):
        lab = mapping.get(str(v or "").strip().lower())
        if lab and lab not in out and str(v) not in drop:
            out.append(lab)
    return ", ".join(out)


def user_stats(t: str) -> tuple[int, int]:
    lines = [l[5:] for l in (t or "").split("\n") if l.startswith("User:")]
    return len(lines), sum(len(l.split()) for l in lines)


def call(method: str, url: str, **kw):
    for attempt in range(5):
        r = sheets._http().request(method, url, timeout=180, **kw)
        if r.status_code in (429, 500, 503):
            time.sleep(8 * (attempt + 1))
            continue
        return sheets._check(r)
    return sheets._check(r)


def write(sid: str, rng: str, rows: list[list], mode: str = "RAW", chunk: int = 400):
    """Rows starting at the top-left cell `rng` (e.g. "Conversations!A1")."""
    tab, cell = rng.rsplit("!", 1)
    start = int("".join(ch for ch in cell if ch.isdigit()))
    letters = "".join(ch for ch in cell if ch.isalpha())
    for i in range(0, len(rows), chunk):
        part = rows[i:i + chunk]
        call("PUT", f"{API}/{sid}/values/{tab}!{letters}{start + i}", params={"valueInputOption": mode},
             json={"values": part})


def rgb(hex_: str) -> dict:
    h = hex_.lstrip("#")
    return {"red": int(h[0:2], 16) / 255, "green": int(h[2:4], 16) / 255, "blue": int(h[4:6], 16) / 255}


def main(sid, calls_pkl, api_json, coded_jsonl, findings_json):
    data = pickle.load(open(calls_pkl, "rb"))
    api = json.load(open(api_json))
    ids = {}
    for c in api:
        started = datetime.fromtimestamp(c["started_at"], IST).strftime("%Y-%m-%d %H:%M")
        ids[(str(c.get("from_number") or ""), started)] = c["id"]
        ids[(str(c.get("from_number") or ""), started, round(float(c.get("duration_secs") or 0)))] = c["id"]
    coded = {}
    for line in Path(coded_jsonl).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "error" not in r:
                coded[r["key"]] = r
    findings = json.load(open(findings_json))

    def expand(text: str) -> str:
        for h in CALL_HEADERS:
            text = text.replace("{AC:" + h + "}", ac(h))
        for h in CONV_HEADERS:
            text = text.replace("{" + h + "}", cv(h))
        return text

    def link(d):
        cid = (ids.get((str(d["phone"] or ""), str(d["started"]), round(float(d["duration_secs"] or 0))))
               or ids.get((str(d["phone"] or ""), str(d["started"]))))
        return f"{BASE}/calls/{cid}" if cid else ""

    # ---------- rows ----------
    from collections import Counter
    key_count = Counter(f'{d["phone"]}|{d["started"]}' for d in data)
    call_rows, conv_rows = [], []
    for d in sorted(data, key=lambda d: str(d["started"])):
        turns, words = user_stats(d["transcript"])
        call_rows.append([str(d["started"]), COHORT.get(d["cohort"] or "", d["cohort"] or "Unknown"),
                          d["theme"] or "Unknown", (d["agent"] or "").replace("genie-", ""), d["status"],
                          d["end_reason"] or "", round(float(d["duration_secs"] or 0)), turns, words, None,
                          link(d)])
        # Two calls to one number in the same minute share the short key: those are
        # coded under a key that also carries the duration.
        key = f'{d["phone"]}|{d["started"]}'
        r = coded.get(f'{key}|{round(float(d["duration_secs"] or 0))}') or (coded.get(key) if key_count[key] == 1 else None)
        if not r:
            continue
        reasons = [REASON_ALIASES.get(x, x) for x in (r.get("cancel_reasons") or [])]
        main = pick(REASONS, REASON_ALIASES.get(r.get("main_reason"), r.get("main_reason")), "No reason stated")
        conv_rows.append([
            str(d["started"]), COHORT.get(d["cohort"] or "", d["cohort"] or "Unknown"), d["theme"] or "Unknown",
            d["video_title"] or "", round(float(d["duration_secs"] or 0)),
            pick(RESPONDENT, r.get("respondent"), "Unclear"), pick(ENGAGEMENT, r.get("engagement"), "No engagement"),
            pick(YESNO, r.get("knows_gyantv"), "Unclear"), pick(AWARE, r.get("aware_of_payment"), "Unclear"),
            main, join(REASONS, reasons, drop=("not_stated",)) or ("" if main == "No reason stated" else main),
            str(r.get("amount_mentioned") or ""), bool(r.get("refund_requested") is True),
            pick(YESNO, r.get("watched_content"), "Unclear"), str(r.get("liked") or ""), str(r.get("disliked") or ""),
            join(DISLIKED, r.get("disliked_tags") or []),
            "; ".join(str(x) for x in (r.get("content_requests") or []) if x),
            join(REQUESTS, r.get("request_categories") or []),
            pick(YESNO, r.get("would_return"), "Not asked"), str(r.get("price_willing") or ""),
            pick(SENTIMENT, r.get("sentiment"), "Neutral"), str(r.get("quote_hi") or ""), str(r.get("quote_en") or ""),
            str(r.get("one_line") or ""), int(r.get("insight_score") or 0),
            bool(r.get("call_audio_problem")), link(d),
            (d["transcript"] or "")[:45000],
        ])
    n_calls, n_conv = len(call_rows), len(conv_rows)
    print(f"{n_calls} calls, {n_conv} coded conversations")

    # ---------- tabs ----------
    TABS = [("Findings", 0), ("Funnel", 11), ("Cancel reasons", 12), ("Content feedback", 13), ("By theme", 14),
            ("Specific feedback", 15), ("Conversations", 16), ("All calls", 17), ("Data sources", 18)]
    meta = call("GET", f"{API}/{sid}", params={"fields": "sheets.properties"})
    have = {s["properties"]["title"]: s["properties"]["sheetId"] for s in meta["sheets"]}
    reqs = []
    if "Findings" not in have:
        first = meta["sheets"][0]["properties"]
        reqs.append({"updateSheetProperties": {"properties": {"sheetId": first["sheetId"], "title": "Findings"},
                                               "fields": "title"}})
        have["Findings"] = first["sheetId"]
    sizes = {"Conversations": (n_conv + 50, len(CONV_HEADERS)), "All calls": (n_calls + 50, len(CALL_HEADERS))}
    for title, wanted_id in TABS:
        if title not in have:
            rows, cols = sizes.get(title, (200, 12))
            reqs.append({"addSheet": {"properties": {"sheetId": wanted_id, "title": title,
                                                     "gridProperties": {"rowCount": rows, "columnCount": cols}}}})
            have[title] = wanted_id
    if reqs:
        call("POST", f"{API}/{sid}:batchUpdate", json={"requests": reqs})
    for title in [t for t, _ in TABS]:
        call("POST", f"{API}/{sid}/values/{title}:clear" if " " not in title else
             f"{API}/{sid}/values/'{title}':clear", json={})

    # ---------- data tabs ----------
    write(sid, "Conversations!A1", [CONV_HEADERS] + conv_rows, "RAW", chunk=250)
    write(sid, "'All calls'!A1", [CALL_HEADERS] + call_rows, "RAW", chunk=2000)
    w = A["User words"]
    outcome = [[f'=IF({w}{i}=0,"User said nothing",IF({w}{i}<5,"Under 5 words","Conversation"))']
               for i in range(2, n_calls + 2)]
    write(sid, f"'All calls'!{A['Outcome']}2", outcome, "USER_ENTERED", chunk=3000)

    cancel = '"*cancelled"'          # both cancelled cohorts
    # ---------- Funnel ----------
    total = "$B$4"
    funnel = [
        ["Half of the people who pick up never say a word", None, None],
        [None, None, None],
        ["Stage", "Calls", "Share of connected calls"],
        ["Calls connected", f"=COUNTA({ac('Call time (IST)')})", f"=IF({total}=0,0,B4/{total})"],
        ["User said nothing", f'=COUNTIF({ac("Outcome")},"User said nothing")', f"=IF({total}=0,0,B5/{total})"],
        ["User said under 5 words", f'=COUNTIF({ac("Outcome")},"Under 5 words")', f"=IF({total}=0,0,B6/{total})"],
        ["Conversations (5+ words)", f'=COUNTIF({ac("Outcome")},"Conversation")', f"=IF({total}=0,0,B7/{total})"],
        ["Answered at least one question", f'=COUNTIF({cv("Engagement")},"Answered several questions")'
                                           f'+COUNTIF({cv("Engagement")},"One real answer")',
         f"=IF({total}=0,0,B8/{total})"],
        ["Answered several questions", f'=COUNTIF({cv("Engagement")},"Answered several questions")',
         f"=IF({total}=0,0,B9/{total})"],
        ["Gave a specific point (insight score 2 or 3)", f'=COUNTIF({cv("Insight score (0-3)")},">=2")',
         f"=IF({total}=0,0,B10/{total})"],
        ["Gave rich, usable feedback (score 3)", f'=COUNTIF({cv("Insight score (0-3)")},3)',
         f"=IF({total}=0,0,B11/{total})"],
        [None, None, None],
        ["Who picked up (coded conversations)", "Conversations", "Share of conversations"],
    ]
    conv_total = f'COUNTA({cv("Call time (IST)")})'
    for label in RESPONDENT.values():
        r = len(funnel) + 1
        funnel.append([label, f'=COUNTIF({cv("Respondent")},A{r})', f"=IF({conv_total}=0,0,B{r}/{conv_total})"])
    funnel += [[None, None, None], ["Do they recognise Gyan TV?", "Conversations", "Share of conversations"]]
    for label in ("Yes", "No", "Unclear"):
        r = len(funnel) + 1
        funnel.append([label, f'=COUNTIF({cv("Knows Gyan TV")},A{r})', f"=IF({conv_total}=0,0,B{r}/{conv_total})"])
    funnel += [[None, None, None], ["Did they know they were paying? (cancelled cohorts)", "Conversations",
                                    "Share of cancelled-cohort conversations"]]
    canc_total = f'COUNTIF({cv("Cohort")},{cancel})'
    for label in AWARE.values():
        r = len(funnel) + 1
        funnel.append([label, f'=COUNTIFS({cv("Cohort")},{cancel},{cv("Aware of payment")},A{r})',
                       f"=IF({canc_total}=0,0,B{r}/{canc_total})"])
    funnel += [[None, None, None], ["Mood of the conversation", "Conversations", "Share of conversations"]]
    for label in SENTIMENT.values():
        r = len(funnel) + 1
        funnel.append([label, f'=COUNTIF({cv("Sentiment")},A{r})', f"=IF({conv_total}=0,0,B{r}/{conv_total})"])
    funnel += [[None, None, None], ["Call quality", "Conversations", "Share of conversations"]]
    r = len(funnel) + 1
    funnel.append(["Could not hear the caller", f'=COUNTIF({cv("Could not hear the caller")},TRUE)',
                   f"=IF({conv_total}=0,0,B{r}/{conv_total})"])
    write(sid, "Funnel!A1", funnel, "USER_ENTERED")
    funnel_rows = len(funnel)

    # ---------- Cancel reasons ----------
    stated = "$B$3"
    rows = [["Why people cancelled, counted over cancelled-cohort conversations", None, None, None, None],
            [None, None, None, None, None],
            ["Conversations with a stated main reason",
             f'=COUNTIF({cv("Cohort")},{cancel})-COUNTIFS({cv("Cohort")},{cancel},{cv("Main reason")},"No reason stated")',
             None, None, None],
            [None, None, None, None, None],
            ["Reason", "Main reason (conversations)", "Share of stated main reasons",
             "Mentioned at all (conversations)", "An example in their words"]]
    first_reason = len(rows) + 1
    for code, label in REASONS.items():
        if code == "not_stated":
            continue
        r = len(rows) + 1
        rows.append([label,
                     f'=COUNTIFS({cv("Cohort")},{cancel},{cv("Main reason")},A{r})',
                     f"=IF({stated}=0,0,B{r}/{stated})",
                     f'=COUNTIFS({cv("Cohort")},{cancel},{cv("All reasons")},"*"&A{r}&"*")',
                     # prefer a call where this was the only reason given, so the quote is about it
                     f'=IFERROR(INDEX(FILTER({cv("Quote (English)")},{cv("All reasons")}=A{r},'
                     f'{cv("Insight score (0-3)")}>=2,{cv("Quote (English)")}<>""),1),'
                     f'IFERROR(INDEX(FILTER({cv("Quote (English)")},{cv("Main reason")}=A{r},'
                     f'{cv("Quote (English)")}<>""),1),""))'])
    last_reason = len(rows)
    rows += [[None] * 5,
             ["Money", "Conversations", "Share of cancelled-cohort conversations", None, None],
             ["Asked for a refund", f'=COUNTIFS({cv("Cohort")},{cancel},{cv("Refund requested")},TRUE)',
              f"=IF({canc_total}=0,0,B{len(rows) + 3}/{canc_total})", None, None],
             ["Mentioned an amount charged", f'=COUNTIFS({cv("Cohort")},{cancel},{cv("Amount mentioned")},"?*")',
              f"=IF({canc_total}=0,0,B{len(rows) + 4}/{canc_total})", None, None],
             [None] * 5,
             ["Would they come back?", "Conversations", "Share of those asked", None, None]]
    asked = f'({canc_total}-COUNTIFS({cv("Cohort")},{cancel},{cv("Would return")},"Not asked"))'
    for label in ("Yes", "Maybe", "No"):
        r = len(rows) + 1
        rows.append([label, f'=COUNTIFS({cv("Cohort")},{cancel},{cv("Would return")},A{r})',
                     f"=IF({asked}=0,0,B{r}/{asked})", None, None])
    write(sid, "'Cancel reasons'!A1", rows, "USER_ENTERED")
    cancel_rows = len(rows)

    # ---------- Content feedback ----------
    rows = [["What people said about the content, all cohorts", None, None, None],
            [None] * 4,
            ["Conversations that said something about content",
             f'=COUNTIFS({cv("Disliked")},"?*")+COUNTIFS({cv("Disliked")},"",{cv("Liked")},"?*")', None, None],
            ["Said what they liked", f'=COUNTIF({cv("Liked")},"?*")', None, None],
            ["Said what they disliked", f'=COUNTIF({cv("Disliked")},"?*")', None, None],
            ["Asked for specific content", f'=COUNTIF({cv("Content requests")},"?*")', None, None],
            [None] * 4,
            ["What they disliked", "Conversations", "Share of those who named a dislike",
             "An example from the coded notes"]]
    dis_total = "$B$5"
    first_dis = len(rows) + 1
    for label in DISLIKED.values():
        r = len(rows) + 1
        rows.append([label, f'=COUNTIF({cv("Disliked tags")},"*"&A{r}&"*")', f"=IF({dis_total}=0,0,B{r}/{dis_total})",
                     f'=IFERROR(INDEX(FILTER({cv("Disliked")},ISNUMBER(SEARCH(A{r},{cv("Disliked tags")})),'
                     f'{cv("Insight score (0-3)")}>=2),1),"")'])
    last_dis = len(rows)
    rows += [[None] * 4,
             ["What they asked for", "Conversations", "Share of those who asked for content",
              "An example of what was asked"]]
    req_total = "$B$6"
    first_req = len(rows) + 1
    for label in REQUESTS.values():
        r = len(rows) + 1
        rows.append([label, f'=COUNTIF({cv("Request categories")},"*"&A{r}&"*")',
                     f"=IF({req_total}=0,0,B{r}/{req_total})",
                     f'=IFERROR(INDEX(FILTER({cv("Content requests")},ISNUMBER(SEARCH(A{r},{cv("Request categories")})),'
                     f'{cv("Content requests")}<>""),1),"")'])
    last_req = len(rows)
    write(sid, "'Content feedback'!A1", rows, "USER_ENTERED")
    content_rows = len(rows)

    # ---------- By theme ----------
    themes = [t for t, _ in Counter(r[2] for r in conv_rows).most_common(14)]
    top_reasons = ["Did not know they would be charged", "Wants a refund", "Expected to earn money from the app",
                   "Content not useful or practical", "Could not find the content they wanted",
                   "No time or not using it", "No money right now"]
    rows = [["Conversations by the theme the user watches", *[None] * (4 + len(top_reasons))],
            [None] * (5 + len(top_reasons)),
            ["Theme", "Calls connected", "Conversations", "Specific feedback (score 2+)",
             "Share of conversations with specific feedback", *[f"Main reason: {x}" for x in top_reasons]],
            [None, None, None, None, None, *top_reasons]]
    for t in themes:
        r = len(rows) + 1
        rows.append([t, f"=COUNTIF({ac('Theme they watch')},$A{r})", f'=COUNTIF({cv("Theme they watch")},$A{r})',
                     f'=COUNTIFS({cv("Theme they watch")},$A{r},{cv("Insight score (0-3)")},">=2")',
                     f"=IF(C{r}=0,0,D{r}/C{r})",
                     *[f'=COUNTIFS({cv("Theme they watch")},$A{r},{cv("Main reason")},{col(6 + i)}$4)'
                       for i in range(len(top_reasons))]])
    write(sid, "'By theme'!A1", rows, "USER_ENTERED")
    theme_rows = len(rows)

    # ---------- Specific feedback (live view) ----------
    keep = ["Call time (IST)", "Cohort", "Theme they watch", "Main reason", "Liked", "Disliked", "Content requests",
            "Quote (English)", "One-line summary", "Insight score (0-3)", "Call link"]
    arr = ",".join(cv(h) for h in keep)
    rows = [["Every conversation with a specific point (insight score 2 or 3), richest first"] + [None] * (len(keep) - 1),
            [None] * len(keep), keep,
            [f'=SORT(FILTER({{{arr}}},{cv("Insight score (0-3)")}>=2),{keep.index("Insight score (0-3)") + 1},FALSE)']]
    write(sid, "'Specific feedback'!A1", rows, "USER_ENTERED")

    # ---------- Findings ----------
    rows = [[findings["headline"], None, None, None],
            [expand(findings["scope"]), None, None, None],
            [None] * 4,
            ["Finding", "Measure", "Value", "How to read it"]]
    fmt_cells = []
    for f in findings["findings"]:
        rows.append([f["finding"], f["metric"], expand(f["formula"]), f["reading"]])
        fmt_cells.append((len(rows), f.get("format", "integer")))
    rows += [[None] * 4, ["What we would do next", None, None, None]]
    for step in findings["next_steps"]:
        rows.append([step, None, None, None])
    rows += [[None] * 4, ["How to use this workbook", None, None, None]]
    for line in findings["guide"]:
        rows.append([line, None, None, None])
    write(sid, "Findings!A1", rows, "USER_ENTERED")
    findings_rows = len(rows)

    # ---------- Data sources ----------
    run_at = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
    ds = [["Written to", "Source", "Object", "Obtained via", "Query or request", "Parameters", "Run at",
           "Requested by", "Row count", "Changes after retrieval", "Notes"],
          [f"All calls!A1:{A['Call link']}{n_calls + 1}", "Gyan TV Genie CRM (call database)",
           f"{BASE}/api/calls/export.xlsx", "Run by Claude",
           "GET /api/calls/export.xlsx?range=all&transport=exotel", "All agents, all outcomes, phone calls only",
           run_at, "Tridev Maurya", n_calls,
           "Sorted by call time. Phone numbers dropped. User turns and words counted from the transcript. "
           "Outcome column is a formula on User words.", "Calls from 26 Sep to 1 Oct 2026."],
          [f"Conversations!A1:{C['Transcript']}{n_conv + 1}", "Gyan TV Genie CRM transcripts, coded by Gemini",
           "gemini-3.1-flash-lite over each transcript (scripts/research_code.py)", "Run by Claude",
           "One codebook prompt per call with 5 or more user words; temperature 0; JSON output",
           "Codebook: respondent, engagement, awareness, cancel reasons, likes, dislikes, content requests, "
           "return intent, sentiment, quote, insight score", run_at, "Tridev Maurya", n_conv,
           "Codes replaced with readable labels. Transcript, cohort, theme, video and duration joined from the "
           "call export.",
           "Model-coded, not hand-coded: spot-check a sample against the transcripts before quoting a number "
           "outside the team."]]
    write(sid, "'Data sources'!A1", ds, "RAW")

    # ---------- formatting ----------
    HDR, HDR_FG, SOFT = "#1F3864", "#FFFFFF", "#EEF2F8"

    def cell_fmt(sheet, r0, r1, c0, c1, fmt, fields):
        return {"repeatCell": {"range": {"sheetId": have[sheet], "startRowIndex": r0, "endRowIndex": r1,
                                         "startColumnIndex": c0, "endColumnIndex": c1},
                               "cell": {"userEnteredFormat": fmt}, "fields": fields}}

    def header(sheet, row, ncols):
        return cell_fmt(sheet, row - 1, row, 0, ncols,
                        {"backgroundColor": rgb(HDR), "wrapStrategy": "WRAP", "verticalAlignment": "MIDDLE",
                         "textFormat": {"bold": True, "foregroundColor": rgb(HDR_FG), "fontFamily": "Arial"}},
                        "userEnteredFormat.backgroundColor,userEnteredFormat.wrapStrategy,"
                        "userEnteredFormat.verticalAlignment,userEnteredFormat.textFormat")

    def title(sheet, ncols):
        return cell_fmt(sheet, 0, 1, 0, ncols, {"textFormat": {"bold": True, "fontSize": 13, "fontFamily": "Arial"}},
                        "userEnteredFormat.textFormat")

    def widths(sheet, spec):
        out = []
        for c0, c1, px in spec:
            out.append({"updateDimensionProperties": {
                "range": {"sheetId": have[sheet], "dimension": "COLUMNS", "startIndex": c0, "endIndex": c1},
                "properties": {"pixelSize": px}, "fields": "pixelSize"}})
        return out

    def number(sheet, r0, r1, c0, c1, pattern, kind="NUMBER"):
        return cell_fmt(sheet, r0, r1, c0, c1, {"numberFormat": {"type": kind, "pattern": pattern}},
                        "userEnteredFormat.numberFormat")

    def freeze(sheet, rows_, cols_=0):
        return {"updateSheetProperties": {"properties": {"sheetId": have[sheet], "gridProperties": {
            "frozenRowCount": rows_, "frozenColumnCount": cols_}},
            "fields": "gridProperties.frozenRowCount,gridProperties.frozenColumnCount"}}

    def wrap(sheet, r0, r1, c0, c1, strategy="WRAP"):
        return cell_fmt(sheet, r0, r1, c0, c1, {"wrapStrategy": strategy, "verticalAlignment": "TOP"},
                        "userEnteredFormat.wrapStrategy,userEnteredFormat.verticalAlignment")

    def note(sheet, row, column, text):
        return {"updateCells": {"rows": [{"values": [{"note": text}]}], "fields": "note",
                                "start": {"sheetId": have[sheet], "rowIndex": row, "columnIndex": column}}}

    reqs = []
    for t, _ in TABS:   # one font everywhere
        reqs.append(cell_fmt(t, 0, None, 0, None, {"textFormat": {"fontFamily": "Arial"}},
                             "userEnteredFormat.textFormat.fontFamily"))
        reqs[-1]["repeatCell"]["range"] = {"sheetId": have[t]}
    # Findings
    reqs += [title("Findings", 4), header("Findings", 4, 4), freeze("Findings", 4),
             wrap("Findings", 4, findings_rows, 0, 4),
             *widths("Findings", [(0, 1, 430), (1, 2, 300), (2, 3, 90), (3, 4, 430)]),
             cell_fmt("Findings", 1, 2, 0, 4, {"textFormat": {"italic": True, "fontFamily": "Arial"}},
                      "userEnteredFormat.textFormat")]
    for r, kind in fmt_cells:
        reqs.append(number("Findings", r - 1, r, 2, 3, "0.0%" if kind == "percent" else "#,##0",
                           "PERCENT" if kind == "percent" else "NUMBER"))
    n_f = len(findings["findings"])
    for label_row in (4 + n_f + 2, 4 + n_f + 2 + len(findings["next_steps"]) + 2):
        reqs.append(cell_fmt("Findings", label_row - 1, label_row, 0, 4,
                             {"backgroundColor": rgb(SOFT), "textFormat": {"bold": True, "fontFamily": "Arial"}},
                             "userEnteredFormat.backgroundColor,userEnteredFormat.textFormat"))
    # Funnel
    reqs += [title("Funnel", 3), *widths("Funnel", [(0, 1, 360), (1, 2, 130), (2, 3, 230)]),
             number("Funnel", 3, funnel_rows, 1, 2, "#,##0"), number("Funnel", 3, funnel_rows, 2, 3, "0.0%", "PERCENT")]
    for i, row in enumerate(funnel, start=1):
        if row[1] in ("Calls", "Conversations"):
            reqs.append(header("Funnel", i, 3))
    # Cancel reasons
    reqs += [title("Cancel reasons", 5), header("Cancel reasons", 5, 5),
             *widths("Cancel reasons", [(0, 1, 320), (1, 2, 150), (2, 3, 170), (3, 4, 170), (4, 5, 520)]),
             number("Cancel reasons", 2, cancel_rows, 1, 2, "#,##0"),
             number("Cancel reasons", 2, cancel_rows, 3, 4, "#,##0"),
             number("Cancel reasons", 4, cancel_rows, 2, 3, "0.0%", "PERCENT"),
             wrap("Cancel reasons", first_reason - 1, last_reason, 4, 5),
             header("Cancel reasons", last_reason + 2, 3), header("Cancel reasons", last_reason + 6, 3)]
    # Content feedback
    reqs += [title("Content feedback", 4), header("Content feedback", first_dis - 1, 4),
             header("Content feedback", first_req - 1, 4),
             *widths("Content feedback", [(0, 1, 340), (1, 2, 130), (2, 3, 230), (3, 4, 560)]),
             number("Content feedback", 2, content_rows, 1, 2, "#,##0"),
             number("Content feedback", first_dis - 1, content_rows, 2, 3, "0.0%", "PERCENT"),
             wrap("Content feedback", first_dis - 1, content_rows, 3, 4)]
    # By theme
    ncol = 5 + len(top_reasons)
    reqs += [title("By theme", ncol), header("By theme", 3, ncol), freeze("By theme", 4, 1),
             *widths("By theme", [(0, 1, 210), (1, 5, 130), (5, ncol, 150)]),
             number("By theme", 4, theme_rows, 1, 4, "#,##0"), number("By theme", 4, theme_rows, 5, ncol, "#,##0"),
             number("By theme", 4, theme_rows, 4, 5, "0.0%", "PERCENT"),
             cell_fmt("By theme", 3, 4, 0, ncol, {"textFormat": {"foregroundColor": rgb("#808080"), "fontSize": 8,
                                                                 "fontFamily": "Arial"}, "wrapStrategy": "CLIP"},
                      "userEnteredFormat.textFormat,userEnteredFormat.wrapStrategy"),
             {"updateDimensionProperties": {"range": {"sheetId": have["By theme"], "dimension": "ROWS",
                                                      "startIndex": 2, "endIndex": 3},
                                            "properties": {"pixelSize": 62}, "fields": "pixelSize"}}]
    # Specific feedback
    reqs += [title("Specific feedback", len(keep)), header("Specific feedback", 3, len(keep)),
             freeze("Specific feedback", 3),
             *widths("Specific feedback", [(0, 1, 125), (1, 3, 140), (3, 4, 210), (4, 7, 260), (7, 9, 330),
                                           (9, 10, 80), (10, 11, 240)]),
             wrap("Specific feedback", 3, None, 0, len(keep))]
    reqs[-1]["repeatCell"]["range"].pop("endRowIndex")
    # Conversations
    nc = len(CONV_HEADERS)
    ci = {h: i for i, h in enumerate(CONV_HEADERS)}
    reqs += [header("Conversations", 1, nc), freeze("Conversations", 1, 1),
             *widths("Conversations", [(0, 1, 125), (1, 4, 150), (4, 5, 80), (5, 10, 150), (10, 11, 260),
                                       (11, 14, 110), (14, 19, 240), (19, 22, 100), (22, 25, 300), (25, 27, 90),
                                       (27, 28, 220), (28, 29, 400)]),
             wrap("Conversations", 1, n_conv + 1, 0, nc, "CLIP"),
             number("Conversations", 1, n_conv + 1, ci["Duration (s)"], ci["Duration (s)"] + 1, "#,##0"),
             {"setBasicFilter": {"filter": {"range": {"sheetId": have["Conversations"], "startRowIndex": 0,
                                                      "endRowIndex": n_conv + 1, "startColumnIndex": 0,
                                                      "endColumnIndex": nc}}}},
             note("Conversations", 0, 0,
                  f"Source: Gyan TV Genie CRM transcripts coded by gemini-3.1-flash-lite, {n_conv} rows, run {run_at}. "
                  "Run by Claude. Details on the Data sources tab.")]
    # All calls
    na = len(CALL_HEADERS)
    reqs += [header("All calls", 1, na), freeze("All calls", 1),
             *widths("All calls", [(0, 1, 125), (1, 3, 170), (3, 6, 130), (6, 9, 90), (9, 10, 150), (10, 11, 260)]),
             number("All calls", 1, n_calls + 1, 6, 9, "#,##0"),
             {"setBasicFilter": {"filter": {"range": {"sheetId": have["All calls"], "startRowIndex": 0,
                                                      "endRowIndex": n_calls + 1, "startColumnIndex": 0,
                                                      "endColumnIndex": na}}}},
             note("All calls", 0, 0,
                  f"Source: Gyan TV Genie CRM (/api/calls/export.xlsx), {n_calls} rows, run {run_at}. Run by Claude. "
                  "Full request on the Data sources tab.")]
    # Data sources
    reqs += [header("Data sources", 1, 11), *widths("Data sources", [(0, 11, 210)]), wrap("Data sources", 1, 3, 0, 11)]
    call("POST", f"{API}/{sid}:batchUpdate", json={"requests": reqs})
    print("written:", f"https://docs.google.com/spreadsheets/d/{sid}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
