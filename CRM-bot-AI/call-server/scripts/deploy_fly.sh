#!/bin/bash
# Deploy to Fly.io. First time: `fly launch --no-deploy` is NOT needed — fly.toml is
# committed; just run this script. Secrets come from .env + gcp-service-account.json.
set -euo pipefail
cd "$(dirname "$0")/../.."            # CRM-bot-AI/ (fly.toml + Dockerfile live here)
APP=$(grep '^app = ' fly.toml | cut -d'"' -f2)
# Python: a venv here or one level up (the checkout may be nested in another folder).
PY=""; for c in "$PWD/venv/bin/python" "$PWD/../venv/bin/python"; do [ -x "$c" ] && PY="$c" && break; done; PY=${PY:-python3}
# Secrets: .env and the service-account file, here or one level up (both gitignored).
ENVF=""; for c in .env ../.env; do [ -f "$c" ] && ENVF="$c" && break; done
SA=""; for c in gcp-service-account.json ../gcp-service-account.json; do [ -f "$c" ] && SA="$c" && break; done

# Never ship code that doesn't even import (a bad edit once took the live app down).
(cd call-server && "$PY" -c "import app.main" >/dev/null 2>&1) || { echo "call-server/app does not import — fix before deploying"; (cd call-server && "$PY" -c "import app.main" 2>&1 | tail -3); exit 1; }

if ! fly status --app "$APP" >/dev/null 2>&1; then
  fly apps create "$APP"
  fly volumes create crm_data --app "$APP" --region sin --size 1 --yes
fi

# Push secrets (only the keys that are set in .env).
SECRETS=()
for k in GOOGLE_API_KEY SARVAM_API_KEY EXOTEL_API_KEY EXOTEL_API_TOKEN EXOTEL_ACCOUNT_SID EXOTEL_SUBDOMAIN DASHBOARD_TOKEN ADMIN_PASSWORD; do
  v=$(grep -E "^$k=" "$ENVF" | head -1 | cut -d= -f2- | sed 's/[[:space:]]*#.*$//' | tr -d '"' | xargs || true)
  [ -n "$v" ] && SECRETS+=("$k=$v")
done
[ -n "$SA" ] && SECRETS+=("GCP_SERVICE_ACCOUNT_JSON=$(cat "$SA")")
fly secrets set --app "$APP" --stage "${SECRETS[@]}"

fly deploy --app "$APP" --ha=false
echo
echo "Dashboard: https://$APP.fly.dev"
