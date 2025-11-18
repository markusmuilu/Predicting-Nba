FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Fix apt keyring issues first
RUN apt-get update || ( \
        apt-get install -y ca-certificates gnupg && apt-get update \
    ) && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "predict_nba.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
