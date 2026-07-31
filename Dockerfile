FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs

RUN uv sync --frozen --no-dev

ENV ANOMALY_CONFIG_DIR=/app/configs
ENV MODEL_CHECKPOINT=/models/last.pt
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "anomaly_diffusion.serving.app:app", \
     "--host", "0.0.0.0", "--port", "8000"]
