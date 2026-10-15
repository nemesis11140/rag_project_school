FROM python:3.12-slim

# libgomp1: torch 运行时依赖的 OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /uvx /usr/local/bin/

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-system \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .
ARG APP_VERSION=development
LABEL org.opencontainers.image.title="rag_project_school" \
      org.opencontainers.image.version=$APP_VERSION

ENV PATH="/opt/venv/bin:$PATH"
EXPOSE 5000
CMD ["python", "main.py"]
