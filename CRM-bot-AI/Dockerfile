# Gyan TV CRM — Genie voice agents. CPU-only (Silero VAD via onnxruntime).
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Repo layout → flat image layout (see call-server/app/paths.py)
COPY call-server/app/ ./app/
COPY calling-agent/agents/ ./agents/
COPY dashboard/ ./static/
COPY calling-agent/fillers/ ./fillers/
COPY call-server/sample_sheets/ ./sample_sheets/
RUN mkdir -p /app/data

ENV HOST=0.0.0.0 PORT=8080 PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["python", "-m", "app.main"]
