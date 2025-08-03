"""
Evaluation script for Strands agent with Python code interpreter on AIME2024 dataset.
"""

import asyncio
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout_engine import RolloutEngine
from strands_math_workflow import StrandsMathWorkflow


def load_aime2024_dataset():
    """Load and prepare the AIME2024 dataset."""
    try:
        # Try to load from registry first
        test_dataset = DatasetRegistry.load_dataset("aime2024", "test")
        if test_dataset is not None:
            return test_dataset
    except Exception:
        pass
    
    # Load directly from HuggingFace
    try:
        from datasets import load_dataset
        print("Loading AIME2024 dataset from HuggingFace...")
        
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        
        def preprocess_fn(example, idx):
            return {
                "question": example["problem"],
                "ground_truth": example["answer"],
                "data_source": "aime2024",
                "idx": idx
            }
        
        dataset = dataset.map(preprocess_fn, with_indices=True)
        
        # Register for future use
        try:
            DatasetRegistry.register_dataset("aime2024", dataset, "test")
        except Exception:
            pass  # Ignore registration errors
            
        return list(dataset)
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Fallback to a sample problem for testing
        return [{
            "question": "What is the sum of all positive integers $n$ such that $1.2n-4.4<5.2$?",
            "ground_truth": "35",
            "data_source": "aime2024",
            "idx": 0
        }]


def evaluate_results(results):
    """Evaluate and print results statistics."""
    total_problems = len(results)
    correct_count = sum(1 for episode in results if episode.is_correct)
    
    print(f"\n🎯 Evaluation Results:")
    print(f"=" * 50)
    print(f"Total problems: {total_problems}")
    print(f"Correct answers: {correct_count}")
    print(f"Accuracy: {correct_count / total_problems * 100:.2f}%")
    
    # Show some examples
    print(f"\n📋 Sample Results:")
    for i, episode in enumerate(results[:3]):
        print(f"\nProblem {i+1}:")
        print(f"Question: {episode.task['question'][:100]}...")
        print(f"Ground Truth: {episode.task['ground_truth']}")
        print(f"Correct: {episode.is_correct}")
        
        # Show the agent's response
        if episode.trajectories:
            agent_name, trajectory = episode.trajectories[0]
            if trajectory.steps:
                final_response = trajectory.steps[-1].model_response or ""
                print(f"Agent Response: {final_response[:200]}...")


async def main():
    """Main evaluation function."""
    print("🧮 AIME2024 Evaluation with Strands Math Agent")
    print("=" * 60)
    
    # Configuration
    n_parallel_tasks = 5  # Start small for testing
    model_name = "Qwen/Qwen3-4B"  # You can change this to your preferred model
    
    # Environment setup
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.makedirs("logs", exist_ok=True)
    
    # Initialize tokenizer and rollout engine
    print(f"🔧 Initializing model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
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
    print("🏗️  Creating workflow engine...")
    engine = AgentWorkflowEngine(
        workflow_cls=StrandsMathWorkflow,
        workflow_args={
            "model_id": model_name,
            "model_config": {
                "temperature": 0.3,  # Lower temperature for math problems
                "max_tokens": 2048
            },
            "strands_agent_config": {},
            "sampling_params": {
                "temperature": 0.3,
                "top_p": 0.95,
                "model": model_name
            },
        },
        rollout_engine=rollout_engine,
        config=None,
        n_parallel_tasks=n_parallel_tasks,
        retry_limit=2,
    )
    
    # Load dataset
    print("📚 Loading AIME2024 dataset...")
    dataset = load_aime2024_dataset()
    
    # For testing, you might want to limit the number of problems
    max_problems = int(os.environ.get("MAX_PROBLEMS", "10"))
    tasks = dataset[:max_problems]
    
    print(f"🎯 Evaluating on {len(tasks)} problems...")
    
    # Run evaluation
    results = await engine.execute_tasks(tasks)
    
    # Evaluate and save results
    evaluate_results(results)
    
    # Save detailed results
    output_file = "logs/strands_math_aime2024_results.json"
    print(f"💾 Saving detailed results to {output_file}")
    
    # Convert episodes to dictionaries for JSON serialization
    json_results = []
    for episode in results:
        episode_dict = episode.to_dict() if hasattr(episode, 'to_dict') else {
            "id": episode.id,
            "task": episode.task,
            "is_correct": episode.is_correct,
            "trajectories": []
        }
        json_results.append(episode_dict)
    
    with open(output_file, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\n✅ Evaluation completed! Results saved to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Strands Math Agent on AIME2024")
    parser.add_argument("--max-problems", type=int, default=10, 
                       help="Maximum number of problems to evaluate (default: 10)")
    parser.add_argument("--parallel-tasks", type=int, default=5,
                       help="Number of parallel tasks (default: 5)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B",
                       help="Model name to use (default: Qwen/Qwen3-4B)")
    parser.add_argument("--base-url", type=str, default="http://localhost:30000/v1",
                       help="API base URL (default: http://localhost:30000/v1)")
    parser.add_argument("--api-key", type=str, default="your-api-key-here",
                       help="API key (default: your-api-key-here)")
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["MAX_PROBLEMS"] = str(args.max_problems)
    
    # Run the evaluation
    asyncio.run(main()) 