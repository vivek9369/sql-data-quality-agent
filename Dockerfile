# ============================================================
# SQL Data Quality Agent — OpenEnv
# Hugging Face Spaces compatible Docker image
# ============================================================
FROM python:3.11-slim

# ------------ Metadata ------------
LABEL maintainer="Vivek Kumar Maurya <vivekmaurya938@gmail.com>"
LABEL description="SQL Data Quality Agent — OpenEnv Environment"
LABEL version="1.0.0"
LABEL org.opencontainers.image.source="https://huggingface.co/spaces"

# ------------ System deps ------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ------------ Working directory ------------
WORKDIR /app

# ------------ Python deps (layer-cached) ------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ------------ Application files ------------
COPY app.py .
COPY environment.py .
COPY tasks.py .
COPY data_generator.py .
COPY reward.py .
COPY inference.py .
COPY openenv.yaml .
COPY pyproject.toml .
COPY README.md .
COPY server/ ./server/

# ------------ Non-root user for security ------------
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# ------------ Runtime env defaults ------------
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Inference will wait for the server to be ready before running
ENV ENV_SERVER_URL=http://localhost:7860

# ------------ Port ------------
EXPOSE 7860

# ------------ Health check ------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:7860/health || exit 1

# ------------ Start server ------------
# Uses uvicorn directly for clean signal handling
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
