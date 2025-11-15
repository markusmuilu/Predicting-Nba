# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install project using pyproject.toml
RUN pip install --no-cache-dir .

# Expose API port
EXPOSE 8000

# Command to start FastAPI
CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
