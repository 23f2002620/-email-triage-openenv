# ── Base image ───────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Environment ──────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# ── System deps (minimal) ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY models.py        .
COPY email_data.py    .
COPY graders.py       .
COPY environment.py   .
COPY main.py          .
COPY inference.py     .
COPY openenv.yaml     .
COPY tests/ ./tests/

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -sf http://localhost:${PORT}/health || exit 1

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE ${PORT}

# ── Run server ────────────────────────────────────────────────────────────────
CMD ["python", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1"]