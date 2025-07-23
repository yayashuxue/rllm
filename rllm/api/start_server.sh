#!/bin/bash

# rLLM Workflow API Server Startup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting rLLM Workflow API Server${NC}"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python is not installed or not in PATH${NC}"
    exit 1
fi

# Check if required files exist
if [ ! -f "rllm/api/api_server.py" ]; then
    echo -e "${RED}❌ api_server.py not found in rllm/api/ directory${NC}"
    echo -e "${YELLOW}💡 Make sure you're running this from the rllm-zero root directory${NC}"
    exit 1
fi

# Install dependencies if requirements file exists
if [ -f "rllm/api/requirements_api.txt" ]; then
    echo -e "${YELLOW}📦 Installing/updating dependencies...${NC}"
    pip install -r rllm/api/requirements_api.txt
else
    echo -e "${YELLOW}⚠️  requirements_api.txt not found, skipping dependency installation${NC}"
fi

# Set default environment variables if not already set
export MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-4B"}
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-"http://localhost:30000/v1"}
export OPENAI_API_KEY=${OPENAI_API_KEY:-"None"}
export N_PARALLEL_TASKS=${N_PARALLEL_TASKS:-"10"}
export HOST=${HOST:-"0.0.0.0"}
export PORT=${PORT:-"8000"}

echo -e "${GREEN}📋 Configuration:${NC}"
echo -e "  Model: ${MODEL_NAME}"
echo -e "  LLM Server: ${OPENAI_BASE_URL}"
echo -e "  Parallel Tasks: ${N_PARALLEL_TASKS}"
echo -e "  Host: ${HOST}"
echo -e "  Port: ${PORT}"

# Create logs directory if it doesn't exist
mkdir -p logs

echo -e "${GREEN}🌐 Starting API server...${NC}"
echo -e "📖 API Documentation will be available at: http://localhost:${PORT}/docs"
echo -e "🔗 Health check endpoint: http://localhost:${PORT}/"

# Start the server
python -m rllm.api.api_server 