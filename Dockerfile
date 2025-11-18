# Use official Python base image
FROM python:3.11-slim

# Ensure Python output is unbuffered
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Install system dependencies safely
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir .

# Expose FastAPI port
EXPOSE 8000

# Default command
CMD ["uvicorn", "predict_nba.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
