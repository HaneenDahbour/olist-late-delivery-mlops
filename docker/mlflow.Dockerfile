FROM python:3.12-slim

WORKDIR /mlflow

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mlflow==3.16.1 \
    boto3==1.43.75 \
    "psycopg[binary]==3.3.4"

EXPOSE 5000

# The actual server command (with credentials interpolated from
# docker-compose.yml's environment) lives in docker-compose.yml's
# `command:` — not baked in here.
