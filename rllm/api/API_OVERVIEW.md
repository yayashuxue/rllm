# rLLM API Overview

## 🎯 **Two Approaches for Serving Workflows**

rLLM provides two ways to serve workflows via HTTP:

1. **Individual Workflow Serving** (Recommended) - Each workflow script can be served independently
2. **Centralized API Server** - Single server that hosts multiple workflow types

### **Individual Workflow Serving** 🌟

The modern, modular approach. Add `serve()` to any workflow script:

```python
from rllm.api import serve

# ... your workflow setup ...
serve(engine, workflow_name="my_workflow", port=8001)
```

**Benefits:**
- ✅ Self-contained and modular
- ✅ No central configuration needed  
- ✅ Easy to deploy and scale independently
- ✅ Simple to add to existing workflow scripts

👉 **See: `SERVE_WORKFLOWS.md` for complete guide**

### **Centralized API Server**

Traditional approach with a single server for all workflows:

```python
python run_api_server.py  # Serves multiple workflow types
```

## 📁 Directory Structure

```
rllm/
├── api/
│   ├── __init__.py              # API package initialization
│   ├── api_server.py            # Main FastAPI server
│   ├── requirements_api.txt     # API dependencies
│   ├── README_api.md           # Detailed API documentation
│   │
│   ├── run_server.py           # Direct server runner
│   ├── start_server.sh         # Bash startup script
│   │
│   ├── client_example.py       # Comprehensive client examples
│   ├── simple_client.py        # Simple test client
│   │
│   ├── Dockerfile              # Container configuration
│   └── docker-compose.yml      # Multi-service deployment
│
└── (other rllm modules...)

# Project Root
run_api_server.py               # Main launcher script
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r rllm/api/requirements_api.txt
```

### 2. Start the Server
```bash
# Recommended: Use the launcher script
python run_api_server.py

# Alternative methods:
python -m rllm.api.api_server
./rllm/api/start_server.sh
python rllm/api/run_server.py
```

### 3. Test the API
```bash
# Simple test
python rllm/api/simple_client.py

# Comprehensive examples
python rllm/api/client_example.py
```

### 4. Access Documentation
- **Interactive docs**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/

## 🔧 Configuration

Set environment variables to customize the server:

```bash
export MODEL_NAME="Qwen/Qwen3-4B"
export OPENAI_BASE_URL="http://localhost:30000/v1"  
export OPENAI_API_KEY="None"
export N_PARALLEL_TASKS="10"
export HOST="0.0.0.0"
export PORT="8000"
```

## 📡 Available Workflows

1. **`critique`**: Multi-agent solver + critic workflow
2. **`single_turn`**: Single agent-environment interaction
3. **`multi_turn`**: Multi-step agent interactions (up to 5 steps)

## 🔗 Key Endpoints

- `POST /execute` - Execute single task synchronously
- `POST /execute_batch` - Execute multiple tasks
- `POST /submit` - Submit task for async execution
- `GET /status/{task_id}` - Check task status
- `GET /results/{task_id}` - Get task results
- `GET /workflows` - List available workflows

## 🐳 Docker Deployment

```bash
# Build and run with docker-compose
cd rllm/api
docker-compose up

# Or build manually
docker build -f rllm/api/Dockerfile -t rllm-api .
docker run -p 8000:8000 rllm-api
```

## 📚 Documentation

- **Full API Guide**: `rllm/api/README_api.md`
- **Client Examples**: `rllm/api/client_example.py`
- **Interactive Docs**: http://localhost:8000/docs (when server is running)

## 🛠️ Development

The API is built with:
- **FastAPI**: Modern, fast web framework
- **Pydantic**: Data validation and settings
- **Uvicorn**: ASGI server for production
- **asyncio**: Asynchronous task execution

For custom workflows, extend the `WorkflowEngine` class in `api_server.py`. 