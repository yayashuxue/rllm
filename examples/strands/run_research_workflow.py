import asyncio
import os

from transformers import AutoTokenizer

os.environ["OTEL_SDK_DISABLED"] = "true"

from research_workflow import StrandsWorkflow

from rllm.engine.rollout_engine import RolloutEngine


async def run_research_workflow():
    """Example: Running StrandsWorkflow for research queries."""
    print("🔬 Running Research Workflow")
    print("=" * 50)
    
    # Configuration
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
    
    # Create research workflow using new simplified constructor
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
    
    # Example research queries
    research_queries = [
        "What are the latest developments in artificial intelligence in 2024?",
        "How does climate change affect ocean temperatures?",
        "What are the health benefits of Mediterranean diet?",
    ]
    
    for i, query in enumerate(research_queries, 1):
        print(f"\n📋 Research Query {i}: {query}")
        print("-" * 60)
        
        # Create task dictionary
        task = {"task": query}
        uid = f"research-{i:03d}"
        
        try:
            # Execute the research workflow
            episode = await workflow(task=task, uid=uid)
            
            print(f"✅ Research completed successfully!")
            print(f"📊 Episode ID: {episode.id}")
            print(f"🎯 Task: {episode.task}")
            print(f"🏆 Reward: {episode.reward}")
            print(f"📈 Number of agent trajectories: {len(episode.trajectories)}")
            
            # Show agent trajectories
            for agent_name, trajectory in episode.trajectories:
                print(f"\n🤖 Agent: {agent_name}")
                print(f"   Steps: {len(trajectory.steps)}")
                if trajectory.steps:
                    last_step = trajectory.steps[-1]
                    if hasattr(last_step, 'chat_completions') and last_step.chat_completions:
                        last_response = last_step.chat_completions[-1].get('content', '')[:200]
                        print(f"   Last response: {last_response}...")
            
        except Exception as e:
            print(f"❌ Error during research: {str(e)}")
        
        print("\n" + "="*60)


async def run_direct_research():
    """Example: Using the research workflow's direct method."""
    print("\n\n🎯 Running Direct Research Method")
    print("=" * 50)
    
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
    
    # Create workflow
    workflow = StrandsWorkflow(
        rollout_engine=rollout_engine,
        model_id=model_name,
        model_config={"temperature": 0.7},
        sampling_params={"temperature": 0.7, "model": model_name}
    )
    
    # Use the direct research method
    query = "What are the benefits of renewable energy?"
    print(f"🔍 Research Query: {query}")
    
    try:
        # Note: This uses the synchronous method
        final_report = workflow.run_research_workflow(query)
        print(f"📄 Final Report: {final_report}")
        
        # Evaluate the report
        reward = workflow.evaluate_report(str(final_report))
        print(f"🏆 Report Reward: {reward}")
        
    except Exception as e:
        print(f"❌ Error during direct research: {str(e)}")


async def simple_research_example():
    """Simple example with just one query to get started quickly."""
    print("\n\n⚡ Simple Research Example")
    print("=" * 40)
    
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
    
    # Create workflow with minimal configuration
    workflow = StrandsWorkflow(
        rollout_engine=rollout_engine,
        model_id=model_name
    )
    
    # Simple query
    query = "What is the current state of renewable energy adoption globally?"
    task = {"task": query}
    uid = "simple-research-001"
    
    print(f"🔍 Query: {query}")
    
    try:
        episode = await workflow(task=task, uid=uid)
        
        print(f"✅ Research completed!")
        print(f"🎯 Task: {episode.task}")
        print(f"🏆 Reward: {episode.reward}")
        
        # Print agent outputs
        for agent_name, trajectory in episode.trajectories:
            print(f"\n🤖 {agent_name}: {len(trajectory.steps)} steps")
            if trajectory.steps:
                # Show the last response from each agent
                last_step = trajectory.steps[-1]
                if hasattr(last_step, 'chat_completions') and last_step.chat_completions:
                    content = last_step.chat_completions[-1].get('content', '')
                    print(f"   Response: {content[:150]}...")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    async def main():
        """Run research workflow examples."""
        print("🌟 Research Workflow Examples")
        print("=" * 70)
        
        # Example 1: Full async workflow execution
        await run_research_workflow()
        
        # Example 2: Direct research method
        await run_direct_research()
        
        # Example 3: Simple research example
        await simple_research_example()
        
        print("\n🎉 All research examples completed!")
    
    # Run the examples
    asyncio.run(main()) 