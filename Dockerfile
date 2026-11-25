FROM python:3.13-slim

# git is needed to clone the repositories being indexed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY repopilot ./repopilot
RUN pip install --no-cache-dir .

# Run as a non-root user; the model cache dir must exist so the Compose volume inherits ownership.
RUN useradd --create-home app \
    && mkdir -p /home/app/.cache/fastembed \
    && chown -R app:app /home/app/.cache
USER app
ENV FASTEMBED_CACHE_PATH=/home/app/.cache/fastembed \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "--factory", "repopilot.api:create_app", "--host", "0.0.0.0", "--port", "8000"]
