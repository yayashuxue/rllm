FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY rllm/api/requirements_api.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy the application code
COPY . .

# Install rLLM in development mode
RUN pip install -e .

# Expose the port
EXPOSE 8000

# Set environment variables with defaults
ENV MODEL_NAME="Qwen/Qwen3-4B"
ENV OPENAI_BASE_URL="http://localhost:30000/v1"
ENV OPENAI_API_KEY="None"
ENV N_PARALLEL_TASKS="10"
ENV HOST="0.0.0.0"
ENV PORT="8000"

# Run the application
CMD ["python", "-m", "rllm.api.api_server"] 