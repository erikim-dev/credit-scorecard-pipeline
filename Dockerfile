# ── Build stage ───────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# Copy only the installed Python packages (no build-essential in runtime)
COPY --from=builder /install /usr/local

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

# Copy project files
COPY src/ src/
COPY api/ api/
COPY app/ app/
COPY models/ models/
COPY monitoring/ monitoring/
COPY sql/ sql/
COPY .streamlit/ .streamlit/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
