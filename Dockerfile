FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Fix Debian keyring problems + install build tools
RUN apt-get update --allow-insecure-repositories || true && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        gnupg \
        dirmngr \
        wget \
        build-essential \
        gcc \
        g++ \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "predict_nba.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
