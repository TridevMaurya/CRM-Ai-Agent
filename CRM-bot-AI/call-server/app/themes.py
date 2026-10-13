"""Content themes: the content sheet's Category column, spelled ~60 different ways
("Part-Time Earnings" / "Part time Earnings" / "PTE"…), folded into one label each.
UPSC's GS subjects (Polity, Economy, Geography…) count as UPSC so the dashboard and
the dialer see one UPSC theme."""

import re

_ALIASES = {
    "Part-Time Earnings": ["part time earnings", "part-time earnings", "part time", "pte"],
    "Government Schemes": ["government schemes", "govt schemes", "govt. schemes"],
    "YouTube": ["youtube"],
    "Instagram": ["instagram", "instgram"],
    "Share Market": ["share market", "share bazaar"],
    "Business": ["business", "business tips"],
    "Self Help": ["self help", "success stories"],
    "English": ["english", "english speaking"],
    "Finance": ["finance", "smart finance planning"],
    "Horror & Crime": ["horror and crime", "horrror and crime", "horror", "horror stories"],
    "Devotion & Mythology": ["devotion and mythology"],
    "Mobile & Tech": ["mobile tips and technology", "science and technology"],
    "UPSC": ["upsc", "polity", "economy", "geography", "internal security", "ethics and integrity",
             "sociology", "environment and ecology", "art and culture", "history optional"],
    "PSC": ["psc", "pcs", "uppsc", "mppsc", "bpsc", "rpsc", "state psc", "state pcs"],
}
_LOOKUP = {a: canon for canon, aliases in _ALIASES.items() for a in aliases}
# Exam-prep themes are a strong intent signal: one UPSC video among ten "Part-Time
# Earnings" ones still makes the user a UPSC user for tagging and dialing.
PRIORITY_THEMES = ("UPSC", "PSC")
# (?<!\S) / (?!\S) instead of \b: Devanagari vowel signs are not \w, so \b misses them.
_MENTION = re.compile(r"(?<!\S)(upsc|psc|pcs|ias|ips|ifs|civil services?|prelims|mains|sarkari|यूपीएससी|पीएससी|सिविल सर्विस)(?=[\s.,!?।]|$)", re.I)
_SPACES = re.compile(r"[\s\-_]+")


def norm_theme(raw: str | None) -> str:
    t = (raw or "").strip()
    if not t:
        return ""
    low = t.lower().replace("&", "and")
    for cand in (low, low.split(" - ")[0]):        # "Devotion & Mythology - Shiv Series"
        key = _SPACES.sub(" ", cand).strip()
        if key in _LOOKUP:
            return _LOOKUP[key]
    return t if t.isupper() else t.title()


def themes_of(categories: list[str] | tuple[str, ...]) -> tuple[str, list[str]]:
    """(primary theme, all themes) from the categories of what someone watched.
    Primary = a priority theme if any video had one, else the most-watched theme."""
    normed = [t for t in (norm_theme(c) for c in categories) if t]
    if not normed:
        return "", []
    all_themes = sorted(set(normed), key=lambda t: (-normed.count(t), t))
    primary = next((t for t in PRIORITY_THEMES if t in all_themes), all_themes[0])
    return primary, all_themes


def theme_mentioned(text: str) -> str:
    """A priority theme the user actually talked about in a transcript / summary."""
    m = _MENTION.search(text or "")
    if not m:
        return ""
    w = m.group(1).lower()
    return "PSC" if w in ("psc", "pcs", "पीएससी") else "UPSC"
