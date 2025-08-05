"""
Simple workflow server for serving individual workflows via HTTP.
Use this to make any workflow script web-accessible with a simple engine.serve() call.
"""

import asyncio
import os
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..engine.agent_workflow_engine import AgentWorkflowEngine


class TaskRequest(BaseModel):
    task: dict[str, Any]
    task_id: str | None = None


class TaskResponse(BaseModel):
    task_id: str
    episode: dict[str, Any]
    status: str = "completed"


class BatchTaskRequest(BaseModel):
    tasks: list[dict[str, Any]]
    task_ids: list[str] | None = None


class BatchTaskResponse(BaseModel):
    task_ids: list[str]
    episodes: list[dict[str, Any]]
    status: str = "completed"


class WorkflowServer:
    """Simple HTTP server for a single workflow."""
    
    def __init__(self, engine: AgentWorkflowEngine, workflow_name: str = "workflow"):
        self.engine = engine
        self.workflow_name = workflow_name
        self.app = self._create_app()
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application."""
        app = FastAPI(
            title=f"rLLM {self.workflow_name.title()} Workflow Server",
            description=f"HTTP API for {self.workflow_name} workflow execution",
            version="1.0.0"
        )
        
        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Health check
        @app.get("/")
        async def health_check():
            return {
                "message": f"rLLM {self.workflow_name} workflow server is running",
                "workflow": self.workflow_name,
                "status": "healthy"
            }
        
        # Workflow info
        @app.get("/info")
        async def workflow_info():
            return {
                "workflow_name": self.workflow_name,
                "workflow_class": self.engine.workflow_cls.__name__,
                "n_parallel_tasks": self.engine.n_parallel_tasks,
                "retry_limit": self.engine.retry_limit
            }
        
        # Execute single task
        @app.post("/execute", response_model=TaskResponse)
        async def execute_task(request: TaskRequest):
            """Execute a single task."""
            try:
                task_id = request.task_id or str(uuid.uuid4())
                
                # Execute the task
                results = await self.engine.execute_tasks([request.task])
                episode = results[0]
                
                return TaskResponse(
                    task_id=task_id,
                    episode=episode.to_dict(),
                    status="completed"
                )
                
            except Exception as e:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Task execution failed: {str(e)}"
                )
        
        # Execute batch of tasks
        @app.post("/execute_batch", response_model=BatchTaskResponse)
        async def execute_batch(request: BatchTaskRequest):
            """Execute multiple tasks in batch."""
            try:
                task_ids = request.task_ids or [str(uuid.uuid4()) for _ in request.tasks]
                
                if len(task_ids) != len(request.tasks):
                    raise HTTPException(
                        status_code=400,
                        detail="Number of task_ids must match number of tasks"
                    )
                
                # Execute the tasks
                results = await self.engine.execute_tasks(request.tasks, task_ids)
                
                # Convert to dictionary format
                episodes = [episode.to_dict() for episode in results]
                
                return BatchTaskResponse(
                    task_ids=task_ids,
                    episodes=episodes,
                    status="completed"
                )
                
            except Exception as e:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Batch execution failed: {str(e)}"
                )
        
        return app
    
    def serve(self, host: str = "0.0.0.0", port: int = 8000, **uvicorn_kwargs):
        """Start the HTTP server."""
        print(f"🚀 Starting {self.workflow_name} workflow server")
        print(f"📍 Host: {host}")
        print(f"🔌 Port: {port}")
        print(f"📖 Health check: http://localhost:{port}/")
        print(f"📋 Workflow info: http://localhost:{port}/info")
        print(f"📖 API docs: http://localhost:{port}/docs")
        print("")
        
        uvicorn.run(self.app, host=host, port=port, **uvicorn_kwargs)


def serve_workflow(
    engine: AgentWorkflowEngine,
    workflow_name: str = "workflow",
    host: str | None = None,
    port: int | None = None,
    **uvicorn_kwargs
):
    """
    Serve a workflow via HTTP.
    
    Args:
        engine: The AgentWorkflowEngine to serve
        workflow_name: Name of the workflow (for display purposes)
        host: Host to bind to (defaults to env var HOST or "0.0.0.0")
        port: Port to bind to (defaults to env var PORT or 8000)
        **uvicorn_kwargs: Additional arguments to pass to uvicorn.run()
    
    Example:
        ```python
        from rllm.api.workflow_server import serve_workflow
        
        # Create your engine
        engine = AgentWorkflowEngine(...)
        
        # Serve it
        serve_workflow(engine, "math", port=8001)
        ```
    """
    # Use environment variables or defaults
    host = host or os.getenv("HOST", "0.0.0.0")
    port = port or int(os.getenv("PORT", "8000"))
    
    server = WorkflowServer(engine, workflow_name)
    server.serve(host=host, port=port, **uvicorn_kwargs)


# Convenience alias
serve = serve_workflow 