import os
import uuid
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer

from ..agents.critique_agent import CritiqueAgent
from ..agents.math_agent import MathAgent
from ..engine.agent_workflow_engine import AgentWorkflowEngine
from ..engine.rollout_engine import RolloutEngine
from ..environments.base.critique_env import CritiqueEnvironment
from ..rewards.reward_fn import math_reward_fn
from ..workflows.critique_workflow import CritiqueWorkflow
from ..workflows.multi_turn_workflow import MultiTurnWorkflow
from ..workflows.single_turn_workflow import SingleTurnWorkflow


# Pydantic models for API
class TaskRequest(BaseModel):
    task: dict[str, Any]
    workflow_type: str = "critique"  # critique, single_turn, multi_turn
    workflow_config: dict[str, Any] | None = None

class TaskBatchRequest(BaseModel):
    tasks: list[dict[str, Any]]
    workflow_type: str = "critique"
    workflow_config: dict[str, Any] | None = None

class TaskResponse(BaseModel):
    task_id: str
    episode: dict[str, Any]
    status: str = "completed"

class TaskBatchResponse(BaseModel):
    task_ids: list[str]
    episodes: list[dict[str, Any]]
    status: str = "completed"

class TaskStatus(BaseModel):
    task_id: str
    status: str  # pending, running, completed, failed
    progress: str | None = None
    result: dict[str, Any] | None = None

class WorkflowEngine:
    """Wrapper class to manage different workflow engines"""
    
    def __init__(self):
        self.engines = {}
        self._initialize_default_engines()
    
    def _initialize_default_engines(self):
        """Initialize default workflow engines"""
        # Set default model and configuration
        model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-4B")
        base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:30000/v1")
        api_key = os.getenv("OPENAI_API_KEY", "None")
        n_parallel_tasks = int(os.getenv("N_PARALLEL_TASKS", "10"))
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        rollout_engine = RolloutEngine(
            engine_name="openai",
            tokenizer=tokenizer,
            openai_kwargs={
                "base_url": base_url,
                "api_key": api_key,
            },
            disable_thinking=False,
        )
        
        # Default sampling params
        default_sampling_params = {
            "temperature": 0.6,
            "top_p": 0.95,
            "model": model_name
        }
        
        # Critique workflow engine
        self.engines["critique"] = AgentWorkflowEngine(
            workflow_cls=CritiqueWorkflow,
            workflow_args={
                "solver_cls": MathAgent,
                "critic_cls": CritiqueAgent,
                "env_cls": CritiqueEnvironment,
                "solver_args": {"accumulate_thinking": False},
                "critic_args": {"accumulate_thinking": False},
                "env_args": {"reward_fn": math_reward_fn},
                "max_prompt_length": 16384,
                "max_response_length": 16384,
                "sampling_params": default_sampling_params,
            },
            rollout_engine=rollout_engine,
            config=None,
            n_parallel_tasks=n_parallel_tasks,
            retry_limit=1,
        )
        
        # Single turn workflow engine
        self.engines["single_turn"] = AgentWorkflowEngine(
            workflow_cls=SingleTurnWorkflow,
            workflow_args={
                "agent_cls": MathAgent,
                "env_cls": CritiqueEnvironment,
                "agent_args": {"accumulate_thinking": False},
                "env_args": {"reward_fn": math_reward_fn},
                "max_prompt_length": 16384,
                "max_response_length": 16384,
                "sampling_params": default_sampling_params,
            },
            rollout_engine=rollout_engine,
            config=None,
            n_parallel_tasks=n_parallel_tasks,
            retry_limit=1,
        )
        
        # Multi turn workflow engine
        self.engines["multi_turn"] = AgentWorkflowEngine(
            workflow_cls=MultiTurnWorkflow,
            workflow_args={
                "agent_cls": MathAgent,
                "env_cls": CritiqueEnvironment,
                "agent_args": {"accumulate_thinking": False},
                "env_args": {"reward_fn": math_reward_fn},
                "max_steps": 5,
                "max_prompt_length": 16384,
                "max_response_length": 16384,
                "sampling_params": default_sampling_params,
            },
            rollout_engine=rollout_engine,
            config=None,
            n_parallel_tasks=n_parallel_tasks,
            retry_limit=1,
        )
    
    def get_engine(self, workflow_type: str) -> AgentWorkflowEngine:
        """Get workflow engine by type"""
        if workflow_type not in self.engines:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
        return self.engines[workflow_type]
    
    def create_custom_engine(self, workflow_type: str, workflow_config: dict[str, Any]) -> AgentWorkflowEngine:
        """Create a custom workflow engine with provided configuration"""
        # This allows users to override default configurations
        # Implementation would depend on specific requirements
        # For now, return the default engine
        return self.get_engine(workflow_type)

# Global state management
workflow_engine = WorkflowEngine()
task_status_store: dict[str, TaskStatus] = {}

# FastAPI app
app = FastAPI(
    title="rLLM Workflow API",
    description="HTTP API for serving agentic workflows with live trajectory collection",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "rLLM Workflow API is running"}

@app.get("/workflows")
async def list_workflows():
    """List available workflow types"""
    return {
        "workflows": list(workflow_engine.engines.keys()),
        "description": {
            "critique": "Multi-agent workflow with solver and critic",
            "single_turn": "Single-turn agent-environment interaction",
            "multi_turn": "Multi-turn agent-environment interaction"
        }
    }

@app.post("/execute", response_model=TaskResponse)
async def execute_task(request: TaskRequest):
    """Execute a single task with the specified workflow"""
    try:
        task_id = str(uuid.uuid4())
        
        # Get the appropriate engine
        engine = workflow_engine.get_engine(request.workflow_type)
        
        # Execute the task
        results = await engine.execute_tasks([request.task])
        episode = results[0]
        
        # Convert to dictionary format
        episode_dict = episode.to_dict()
        
        return TaskResponse(
            task_id=task_id,
            episode=episode_dict,
            status="completed"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")

@app.post("/execute_batch", response_model=TaskBatchResponse)
async def execute_batch(request: TaskBatchRequest):
    """Execute multiple tasks in batch with the specified workflow"""
    try:
        task_ids = [str(uuid.uuid4()) for _ in request.tasks]
        
        # Get the appropriate engine
        engine = workflow_engine.get_engine(request.workflow_type)
        
        # Execute the tasks
        results = await engine.execute_tasks(request.tasks, task_ids)
        
        # Convert to dictionary format
        episodes = [episode.to_dict() for episode in results]
        
        return TaskBatchResponse(
            task_ids=task_ids,
            episodes=episodes,
            status="completed"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch execution failed: {str(e)}")

async def execute_task_async(task_id: str, task: dict[str, Any], workflow_type: str):
    """Execute task asynchronously and update status"""
    try:
        # Update status to running
        task_status_store[task_id].status = "running"
        task_status_store[task_id].progress = "Executing workflow..."
        
        # Get the appropriate engine
        engine = workflow_engine.get_engine(workflow_type)
        
        # Execute the task
        results = await engine.execute_tasks([task])
        episode = results[0]
        
        # Update status to completed
        task_status_store[task_id].status = "completed"
        task_status_store[task_id].result = episode.to_dict()
        
    except Exception as e:
        # Update status to failed
        task_status_store[task_id].status = "failed"
        task_status_store[task_id].result = {"error": str(e)}

@app.post("/submit")
async def submit_task(request: TaskRequest, background_tasks: BackgroundTasks):
    """Submit a task for asynchronous execution"""
    task_id = str(uuid.uuid4())
    
    # Initialize task status
    task_status_store[task_id] = TaskStatus(
        task_id=task_id,
        status="pending",
        progress="Task submitted, waiting to start..."
    )
    
    # Add to background tasks
    background_tasks.add_task(
        execute_task_async,
        task_id,
        request.task,
        request.workflow_type
    )
    
    return {"task_id": task_id, "status": "submitted"}

@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get the status of a submitted task"""
    if task_id not in task_status_store:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task_status_store[task_id]

@app.get("/results/{task_id}")
async def get_task_results(task_id: str):
    """Get the results of a completed task"""
    if task_id not in task_status_store:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_status = task_status_store[task_id]
    
    if task_status.status != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Task is not completed. Current status: {task_status.status}"
        )
    
    return task_status.result

@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task and its results"""
    if task_id not in task_status_store:
        raise HTTPException(status_code=404, detail="Task not found")
    
    del task_status_store[task_id]
    return {"message": f"Task {task_id} deleted successfully"}

@app.get("/tasks")
async def list_tasks():
    """List all tasks and their statuses"""
    return {
        "tasks": [
            {
                "task_id": task_id,
                "status": status.status,
                "progress": status.progress
            }
            for task_id, status in task_status_store.items()
        ]
    }

if __name__ == "__main__":
    import uvicorn
    
    # Configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    print("port:", port)
    
    uvicorn.run(app, host=host, port=port) 