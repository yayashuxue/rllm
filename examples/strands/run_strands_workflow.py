import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer

os.environ["OTEL_SDK_DISABLED"] = "true"

from strands_workflow import StrandsWorkflow

from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout_engine import RolloutEngine


async def run_with_agent_workflow_engine():
    """Example 1: Using AgentWorkflowEngine for parallel execution."""
    print("🚀 Running StrandsWorkflow with AgentWorkflowEngine (Parallel)")
    print("=" * 70)
    
    # Configuration
    n_parallel_tasks = 5  # Reduced for demo
    model_name = "Qwen/Qwen3-4B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
 
    # Create rollout engine
    rollout_engine = RolloutEngine(
        engine_name="openai",
        tokenizer=tokenizer,
        openai_kwargs={
            "base_url": "http://localhost:30000/v1",  # Adjust to your API endpoint
            "api_key": "your-api-key-here",  # Set your API key
        },
        disable_thinking=False,
    )
    
    # Create workflow engine
    engine = AgentWorkflowEngine(
        workflow_cls=StrandsWorkflow,
        workflow_args={
            "model_id": model_name,
            "model_config": {
                "temperature": 0.7,
                "max_tokens": 1000
            },
            "strands_agent_config": {},
            "sampling_params": {
                "temperature": 0.7, 
                "top_p": 0.95, 
                "model": model_name
            },
            "max_prompt_length": 4096,
            "max_response_length": 2048,
        },
        rollout_engine=rollout_engine,
        config=None,
        n_parallel_tasks=n_parallel_tasks,
        retry_limit=1,
    )
    
    # Load tasks - using a small math dataset for demo
    tasks = [{"task": "What is the capital of France?"}]

    results = await engine.execute_tasks(tasks)

    print(results)
        


async def run_direct_execution():
    """Example 2: Direct workflow execution without AgentWorkflowEngine."""
    print("\n\n🎯 Running StrandsWorkflow directly (Sequential)")
    print("=" * 70)
    
    # Create rollout engine
    model_name = "Qwen/Qwen3-4B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    rollout_engine = RolloutEngine(
        engine_name="openai",
        tokenizer=tokenizer,
        openai_kwargs={
            "base_url": "http://localhost:30000/v1",
            "api_key": "your-api-key-here",
        },
        disable_thinking=False,
    )
    

    workflow = StrandsWorkflow(
        rollout_engine=rollout_engine,
        model_id=model_name,
        model_config={
            "temperature": 0.7,
            "max_tokens": 1000
        },
        strands_agent_config={},
        sampling_params={
            "temperature": 0.7,
            "top_p": 0.95,
            "model": model_name
        }
    )
    
    tasks = [{"task": "What is the capital of France?"}]
    
    if not tasks:
        print("❌ No tasks loaded")
        return
        
    task = tasks[0]
    print(f"📝 Task: {task['task'][:100]}...")
    print()

    episode = await workflow(task=task, uid="direct-example-001")
    print(episode)


if __name__ == "__main__":
    import os
    
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    async def main():
        """Run both examples."""
        print("🌟 StrandsWorkflow Examples")
        print("=" * 70)
        
        # Example 1: Parallel execution with AgentWorkflowEngine
        await run_with_agent_workflow_engine()
        
        # Example 2: Direct execution
        await run_direct_execution()
        
        print("\n🎉 All examples completed!")
    
    # Run the examples
    asyncio.run(main()) 