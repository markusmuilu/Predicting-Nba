# Use official Python base image
FROM python:3.11-slim

# Ensure Python output is unbuffered (shows logs instantly)
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies (if numpy/scipy needs them)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies via pyproject.toml
RUN pip install --no-cache-dir .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI by default
CMD ["uvicorn", "predict_nba.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
