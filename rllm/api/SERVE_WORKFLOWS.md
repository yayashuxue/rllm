# Serving Individual Workflows via HTTP

This guide shows you how to make any rLLM workflow servable via HTTP with a simple `serve()` function call.

## 🎯 **Overview**

Instead of having a centralized API server that loads all workflows, you can now serve individual workflows directly from their own scripts. This approach offers:

- **Self-contained workflows**: Each workflow script becomes independently servable
- **No central configuration**: No need to modify a central API server when adding new workflows
- **Easy deployment**: Take any existing workflow script and make it web-accessible
- **Modular and flexible**: Each workflow can have its own configuration and endpoints

## 🚀 **Quick Start**

### 1. **Basic Usage**

Add these lines to any workflow script:

```python
from rllm.api import serve

# ... your workflow setup code ...

# Serve the workflow
serve(engine, workflow_name="my_workflow", port=8001)
```

### 2. **Complete Example**

Here's how to modify an existing workflow script:

```python
import argparse
from rllm.api import serve
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
# ... other imports ...

def create_engine():
    """Create and return your workflow engine."""
    # Your engine setup code here
    engine = AgentWorkflowEngine(
        workflow_cls=YourWorkflow,
        workflow_args={...},
        rollout_engine=rollout_engine,
        # ... other config ...
    )
    return engine

def run_evaluation():
    """Run the workflow evaluation."""
    engine = create_engine()
    # Your evaluation code here

def run_server(port=8001):
    """Serve the workflow via HTTP."""
    engine = create_engine()
    serve(engine, workflow_name="my_workflow", port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["eval", "serve"])
    parser.add_argument("--port", type=int, default=8001)
    
    args = parser.parse_args()
    
    if args.mode == "eval":
        run_evaluation()
    elif args.mode == "serve":
        run_server(port=args.port)
```

## 📖 **API Reference**

### `serve(engine, workflow_name, host, port, **kwargs)`

Serve a workflow via HTTP.

**Parameters:**
- `engine`: `AgentWorkflowEngine` - The workflow engine to serve
- `workflow_name`: `str` - Display name for the workflow (default: "workflow")
- `host`: `str` - Host to bind to (default: env var `HOST` or "0.0.0.0")
- `port`: `int` - Port to bind to (default: env var `PORT` or 8000)
- `**kwargs`: Additional arguments passed to `uvicorn.run()`

**Example:**
```python
serve(engine, "math", port=8001, reload=True)
```

## 🌐 **HTTP Endpoints**

When you serve a workflow, it automatically provides these endpoints:

### **Health Check**
```
GET /
```
Returns server status and workflow info.

### **Workflow Info**
```
GET /info
```
Returns detailed workflow configuration.

### **Execute Single Task**
```
POST /execute
```

**Request:**
```json
{
  "task": {
    "question": "What is 2 + 2?",
    "ground_truth": "4",
    "idx": 0,
    "data_source": "math"
  },
  "task_id": "optional-custom-id"
}
```

**Response:**
```json
{
  "task_id": "generated-or-custom-id",
  "episode": {
    "id": "episode-id",
    "task": {...},
    "is_correct": true,
    "trajectories": {...}
  },
  "status": "completed"
}
```

### **Execute Batch**
```
POST /execute_batch
```

**Request:**
```json
{
  "tasks": [
    {"question": "What is 2 + 2?", ...},
    {"question": "What is 3 + 3?", ...}
  ],
  "task_ids": ["optional", "custom-ids"]
}
```

**Response:**
```json
{
  "task_ids": ["id1", "id2"],
  "episodes": [{...}, {...}],
  "status": "completed"
}
```

### **API Documentation**
```
GET /docs
```
Interactive Swagger UI documentation.

## 🛠️ **Examples**

### **Math Workflow**

```bash
# Run evaluation
python examples/workflow/math/run_math_workflow.py eval

# Serve via HTTP
python examples/workflow/math/run_math_workflow.py serve --port 8001

# Test the server
python examples/workflow/math/test_math_server.py
```

### **Creating Your Own Servable Workflow**

1. **Start with any workflow script:**

```python
# my_workflow.py
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
# ... setup your workflow ...

engine = AgentWorkflowEngine(...)
```

2. **Add serving capability:**

```python
from rllm.api import serve

def main():
    # ... create your engine ...
    
    # Add command line args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    if args.serve:
        serve(engine, "my_workflow", port=args.port)
    else:
        # Run normal workflow execution
        results = asyncio.run(engine.execute_tasks(tasks))
```

3. **Serve it:**

```bash
python my_workflow.py --serve --port 8002
```

## 🔧 **Configuration**

### **Environment Variables**

- `HOST`: Default host to bind to (default: "0.0.0.0")
- `PORT`: Default port to use (default: 8000)

### **Custom Configuration**

You can customize the server behavior:

```python
from rllm.api.workflow_server import WorkflowServer

# Create custom server
server = WorkflowServer(engine, "my_workflow")

# Customize the FastAPI app
server.app.add_middleware(...)  # Add custom middleware
server.app.mount("/static", ...)  # Add static files

# Serve with custom settings
server.serve(host="localhost", port=8003, reload=True)
```

## 🐳 **Docker Deployment**

### **Basic Dockerfile**

```dockerfile
FROM python:3.10

WORKDIR /app
COPY . .
RUN pip install -e .

EXPOSE 8000
CMD ["python", "my_workflow.py", "--serve"]
```

### **Docker Compose**

```yaml
version: '3.8'
services:
  my-workflow:
    build: .
    ports:
      - "8001:8000"
    environment:
      - HOST=0.0.0.0
      - PORT=8000
```

## 🧪 **Testing Your Served Workflow**

### **Python Client**

```python
import requests

def test_workflow(base_url="http://localhost:8001"):
    # Health check
    response = requests.get(f"{base_url}/")
    print(response.json())
    
    # Execute task
    task = {"question": "What is 2 + 2?", "ground_truth": "4"}
    response = requests.post(
        f"{base_url}/execute",
        json={"task": task}
    )
    print(response.json())

test_workflow()
```

### **cURL**

```bash
# Health check
curl http://localhost:8001/

# Execute task
curl -X POST http://localhost:8001/execute \
  -H "Content-Type: application/json" \
  -d '{
    "task": {
      "question": "What is 2 + 2?",
      "ground_truth": "4"
    }
  }'
```

## 🎛️ **Advanced Usage**

### **Multiple Workflows on Different Ports**

```python
# Terminal 1
serve(math_engine, "math", port=8001)

# Terminal 2  
serve(code_engine, "code", port=8002)

# Terminal 3
serve(custom_engine, "custom", port=8003)
```

### **Custom Endpoints**

```python
from rllm.api.workflow_server import WorkflowServer

server = WorkflowServer(engine, "my_workflow")

# Add custom endpoint
@server.app.get("/custom")
async def custom_endpoint():
    return {"custom": "data"}

server.serve(port=8001)
```

## 🔍 **Best Practices**

1. **Use descriptive workflow names**: Make it clear what each workflow does
2. **Set appropriate ports**: Avoid conflicts with other services
3. **Add error handling**: Gracefully handle workflow failures
4. **Monitor resource usage**: Adjust `n_parallel_tasks` based on your system
5. **Use environment variables**: Make configuration flexible
6. **Add logging**: Monitor workflow execution and performance

## 🚦 **Production Considerations**

1. **Use a reverse proxy** (nginx) for production deployments
2. **Add authentication** if serving over public networks
3. **Implement rate limiting** to prevent abuse
4. **Set up monitoring** and health checks
5. **Use proper logging** with structured formats
6. **Consider scaling** with multiple instances behind a load balancer

## 📚 **More Examples**

Check out the examples directory for more servable workflow scripts:

- `examples/workflow/math/run_math_workflow.py` - Math problem solving
- `examples/workflow/code/run_code_workflow.py` - Code generation and execution

## 🤝 **Migration from Central API**

If you're currently using the central API server (`rllm.api.api_server`), you can easily migrate:

1. **Extract workflow config** from the central server
2. **Create individual scripts** for each workflow type
3. **Add serve() calls** to each script
4. **Deploy independently** on different ports or servers

This gives you more flexibility and easier maintenance! 