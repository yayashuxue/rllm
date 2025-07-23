# rLLM Workflow Web API

A FastAPI-based web service that serves agentic workflows from the rLLM framework. This API allows you to execute workflows via HTTP requests and collect trajectories live.

## Features

- **Multiple Workflow Types**: Support for critique, single-turn, and multi-turn workflows
- **Synchronous & Asynchronous Execution**: Execute tasks immediately or submit for background processing
- **Batch Processing**: Execute multiple tasks simultaneously
- **Live Status Tracking**: Monitor task progress and collect results
- **RESTful API**: Clean HTTP interface with automatic documentation

## Quick Start

### 1. Installation

```bash
# Install API dependencies (from project root)
pip install -r rllm/api/requirements_api.txt

# Ensure rLLM is properly installed
pip install -e .
```

### 2. Environment Configuration

Set the following environment variables (optional):

```bash
export MODEL_NAME="Qwen/Qwen3-4B"                    # Model to use
export OPENAI_BASE_URL="http://localhost:30000/v1"   # LLM server endpoint
export OPENAI_API_KEY="None"                         # API key (if required)
export N_PARALLEL_TASKS="10"                         # Number of parallel tasks
export HOST="0.0.0.0"                                # Server host
export PORT="8000"                                    # Server port
```

### 3. Start the Server

```bash
# From project root directory
python run_api_server.py

# Or using the module
python -m rllm.api.api_server

# Or using the start script (from project root)
./rllm/api/start_server.sh
```

The API will be available at `http://localhost:8000` with automatic documentation at `http://localhost:8000/docs`.

## API Endpoints

### Core Endpoints

- `GET /` - Health check
- `GET /workflows` - List available workflow types
- `POST /execute` - Execute a single task synchronously
- `POST /execute_batch` - Execute multiple tasks in batch
- `POST /submit` - Submit a task for asynchronous execution
- `GET /status/{task_id}` - Get task status
- `GET /results/{task_id}` - Get task results
- `GET /tasks` - List all tasks
- `DELETE /tasks/{task_id}` - Delete a task

## Usage Examples

### 1. List Available Workflows

```bash
curl -X GET "http://localhost:8000/workflows"
```

Response:
```json
{
  "workflows": ["critique", "single_turn", "multi_turn"],
  "description": {
    "critique": "Multi-agent workflow with solver and critic",
    "single_turn": "Single-turn agent-environment interaction",
    "multi_turn": "Multi-turn agent-environment interaction"
  }
}
```

### 2. Execute a Single Task (Synchronous)

```bash
curl -X POST "http://localhost:8000/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "task": {
      "question": "What is 2 + 2?",
      "ground_truth": "4",
      "idx": 0,
      "data_source": "math"
    },
    "workflow_type": "critique"
  }'
```

### 3. Execute Multiple Tasks (Batch)

```bash
curl -X POST "http://localhost:8000/execute_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "tasks": [
      {
        "question": "What is 2 + 2?",
        "ground_truth": "4",
        "idx": 0,
        "data_source": "math"
      },
      {
        "question": "What is 3 * 4?",
        "ground_truth": "12",
        "idx": 1,
        "data_source": "math"
      }
    ],
    "workflow_type": "single_turn"
  }'
```

### 4. Submit Task for Asynchronous Execution

```bash
# Submit task
curl -X POST "http://localhost:8000/submit" \
  -H "Content-Type: application/json" \
  -d '{
    "task": {
      "question": "Solve x^2 + 3x + 2 = 0",
      "ground_truth": "x = -1, x = -2",
      "idx": 0,
      "data_source": "math"
    },
    "workflow_type": "multi_turn"
  }'

# Response: {"task_id": "uuid-string", "status": "submitted"}

# Check status
curl -X GET "http://localhost:8000/status/{task_id}"

# Get results when completed
curl -X GET "http://localhost:8000/results/{task_id}"
```

## Python Client Examples

### Basic Usage

```python
import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def execute_task(task, workflow_type="critique"):
    """Execute a single task synchronously"""
    response = requests.post(
        f"{BASE_URL}/execute",
        json={
            "task": task,
            "workflow_type": workflow_type
        }
    )
    return response.json()

def submit_task_async(task, workflow_type="critique"):
    """Submit task for asynchronous execution"""
    response = requests.post(
        f"{BASE_URL}/submit",
        json={
            "task": task,
            "workflow_type": workflow_type
        }
    )
    return response.json()["task_id"]

def get_task_status(task_id):
    """Get task status"""
    response = requests.get(f"{BASE_URL}/status/{task_id}")
    return response.json()

def get_task_results(task_id):
    """Get task results"""
    response = requests.get(f"{BASE_URL}/results/{task_id}")
    return response.json()

# Example usage
task = {
    "question": "A triangle has sides of length 3, 4, and 5. What is its area?",
    "ground_truth": "6",
    "idx": 0,
    "data_source": "math"
}

# Synchronous execution
result = execute_task(task, "critique")
print("Synchronous result:", result["episode"]["is_correct"])

# Asynchronous execution
task_id = submit_task_async(task, "single_turn")
print(f"Submitted task: {task_id}")

# Poll for completion
import time
while True:
    status = get_task_status(task_id)
    print(f"Status: {status['status']}")
    
    if status["status"] == "completed":
        results = get_task_results(task_id)
        print("Async result:", results["is_correct"])
        break
    elif status["status"] == "failed":
        print("Task failed:", status.get("result", {}).get("error"))
        break
    
    time.sleep(1)
```

### Batch Processing

```python
import requests

def execute_batch(tasks, workflow_type="critique"):
    """Execute multiple tasks in batch"""
    response = requests.post(
        f"{BASE_URL}/execute_batch",
        json={
            "tasks": tasks,
            "workflow_type": workflow_type
        }
    )
    return response.json()

# Prepare multiple tasks
tasks = [
    {
        "question": f"What is {i} + {i+1}?",
        "ground_truth": str(2*i + 1),
        "idx": i,
        "data_source": "math"
    }
    for i in range(1, 6)
]

# Execute batch
results = execute_batch(tasks, "single_turn")

# Process results
for i, episode in enumerate(results["episodes"]):
    print(f"Task {i}: Correct = {episode['is_correct']}")
```

## Response Format

### Episode Structure

Each executed task returns an `Episode` object with the following structure:

```json
{
  "id": "unique-episode-id",
  "task": {
    "question": "input question",
    "ground_truth": "expected answer",
    "idx": 0,
    "data_source": "math"
  },
  "termination_reason": "env_done",
  "is_correct": true,
  "trajectories": {
    "solver": {
      "task": {...},
      "steps": [
        {
          "chat_completions": [...],
          "observation": "...",
          "thought": "...",
          "action": "...",
          "model_response": "...",
          "info": {...},
          "reward": 1.0,
          "done": false,
          "mc_return": 1.0
        }
      ],
      "reward": 1.0
    },
    "critic": {
      // Similar structure for critic agent (if applicable)
    }
  }
}
```

## Configuration

### Workflow Types

1. **critique**: Multi-agent workflow with a solver and critic agent
2. **single_turn**: Single interaction between agent and environment
3. **multi_turn**: Multiple turns of agent-environment interaction (up to 5 steps)

### Environment Variables

- `MODEL_NAME`: HuggingFace model identifier
- `OPENAI_BASE_URL`: LLM inference server endpoint
- `OPENAI_API_KEY`: Authentication key for LLM server
- `N_PARALLEL_TASKS`: Number of concurrent task executions
- `HOST`: Server bind address
- `PORT`: Server port

## Production Considerations

1. **Authentication**: Add proper authentication mechanisms
2. **Rate Limiting**: Implement rate limiting for API endpoints
3. **Persistence**: Use a proper database for task storage instead of in-memory dict
4. **Monitoring**: Add logging and metrics collection
5. **Scaling**: Consider using Redis/Celery for distributed task execution
6. **CORS**: Configure CORS policy appropriately for your frontend

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure the model specified in `MODEL_NAME` is available
2. **Connection Errors**: Verify the LLM server is running at `OPENAI_BASE_URL`
3. **Memory Issues**: Reduce `N_PARALLEL_TASKS` if running out of memory
4. **Timeout Errors**: Increase timeout settings for long-running tasks

### Debug Mode

Run with debug logging:

```bash
# From project root
uvicorn rllm.api.api_server:app --host 0.0.0.0 --port 8000 --log-level debug
```

## API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation powered by Swagger UI. 