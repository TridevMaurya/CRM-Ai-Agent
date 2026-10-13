"""Where things live. Two layouts are supported without any configuration:

    repo layout (this GitHub repo)              container layout (Dockerfile)
    CRM-bot-AI/                                 /app/
      dashboard/        ← STATIC_DIR              static/
      calling-agent/agents/   ← AGENTS_DIR        agents/
      calling-agent/fillers/  ← FILLERS_DIR       fillers/
      call-server/app/  ← this package            app/
      .env, gcp-service-account.json              (Fly secrets)

Each can also be forced with an environment variable of the same name.
`.env` and the Google service-account JSON are looked for in the package root,
the project root and one level above it (so a checkout nested in another folder
can keep its secrets outside the repo)."""

from __future__ import annotations

import os
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent            # call-server/  or /app
PROJECT_ROOT = PKG_ROOT.parent if (PKG_ROOT.parent / "dashboard").is_dir() else PKG_ROOT
_SECRET_DIRS = (PKG_ROOT, PROJECT_ROOT, PROJECT_ROOT.parent)


def _dir(env: str, *candidates: Path) -> Path:
    if os.getenv(env):
        return Path(os.environ[env]).expanduser().resolve()
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[-1]


def find_file(name: str) -> Path | None:
    for d in _SECRET_DIRS:
        if (d / name).is_file():
            return d / name
    return None


STATIC_DIR = _dir("STATIC_DIR", PKG_ROOT / "static", PROJECT_ROOT / "dashboard")
AGENTS_DIR = _dir("AGENTS_DIR", PKG_ROOT / "agents", PROJECT_ROOT / "calling-agent" / "agents")
FILLERS_DIR = _dir("FILLERS_DIR", PKG_ROOT / "fillers", PROJECT_ROOT / "calling-agent" / "fillers")
ENV_FILE = find_file(".env")
SA_FILE = find_file("gcp-service-account.json")
