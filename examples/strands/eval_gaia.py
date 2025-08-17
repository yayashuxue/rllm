"""
Reusable GAIA evaluation script for Strands agents.

This script provides a standalone GAIA evaluation that can be used by any Strands example.
It handles GAIA dataset loading, answer validation, and result reporting.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer

from rllm.data.utils import load_dataset
from rllm.data.dataset_types import TestDataset
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout_engine import RolloutEngine
from rllm.integrations.strands import RLLMModel, StrandsAgent


class GAIAWorkflow:
    """Workflow for GAIA evaluation with Strands agents."""
    
    def __init__(self, agent: StrandsAgent):
        self.agent = agent
        
    async def run_single_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single GAIA task and return results."""
        question = task.get("Question", "")
        ground_truth = task.get("Final answer", "")
        task_id = task.get("task_id", "unknown")
        
        print(f"\n=== GAIA Task {task_id} ===")
        print(f"Question: {question[:200]}...")
        
        try:
            # Reset agent trajectory for new task
            self.agent.reset_trajectory(task=task)
            
            # Run the task
            result = await self.agent.invoke_async(question)
            
            # Extract final answer from result
            final_answer = self._extract_final_answer(result)
            
            # Validate answer
            is_correct = self._validate_answer(final_answer, ground_truth)
            
            return {
                "task_id": task_id,
                "question": question,
                "predicted_answer": final_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "trajectory": self.agent.trajectory,
                "num_steps": len(self.agent.trajectory.steps)
            }
            
        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
            return {
                "task_id": task_id,
                "question": question,
                "predicted_answer": "",
                "ground_truth": ground_truth,
                "is_correct": False,
                "error": str(e),
                "trajectory": None,
                "num_steps": 0
            }
    
    def _extract_final_answer(self, result: Any) -> str:
        """Extract final answer from agent result."""
        if hasattr(result, 'message') and result.message:
            if hasattr(result.message, 'content'):
                if isinstance(result.message.content, list):
                    # Extract text from content blocks
                    text_content = ""
                    for block in result.message.content:
                        if isinstance(block, dict) and 'text' in block:
                            text_content += block['text']
                    return text_content.strip()
                else:
                    return str(result.message.content).strip()
        return str(result).strip() if result else ""
    
    def _validate_answer(self, predicted: str, ground_truth: str) -> bool:
        """Validate predicted answer against ground truth."""
        if not predicted or not ground_truth:
            return False
        
        # Simple string matching for now
        # TODO: Implement more sophisticated answer validation
        predicted_clean = predicted.lower().strip()
        ground_truth_clean = ground_truth.lower().strip()
        
        return predicted_clean == ground_truth_clean


def load_gaia_dataset(subset: str = "validation", limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load GAIA dataset using rLLM's data loading infrastructure."""
    try:
        # Use rLLM's standardized data loading
        data = load_dataset(TestDataset.Web.GAIA)
        
        if isinstance(data, list):
            dataset = data
        else:
            # Convert to list if needed
            dataset = list(data)
        
        # Apply limit if specified
        if limit:
            dataset = dataset[:limit]
        
        print(f"Loaded {len(dataset)} GAIA tasks")
        return dataset
        
    except Exception as e:
        print(f"Error loading GAIA dataset: {e}")
        print("Make sure GAIA data is downloaded. Run: python scripts/data/download_gaia.py")
        return []


async def evaluate_gaia(
    agent: StrandsAgent,
    limit: Optional[int] = None,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate a Strands agent on GAIA dataset.
    
    Args:
        agent: StrandsAgent instance to evaluate
        limit: Maximum number of tasks to evaluate (None for all)
        output_file: Path to save detailed results (optional)
    
    Returns:
        Dictionary with evaluation results
    """
    print("=== GAIA Evaluation for Strands Agent ===")
    
    # Load dataset
    dataset = load_gaia_dataset(limit=limit)
    if not dataset:
        return {"error": "Failed to load GAIA dataset"}
    
    # Create workflow
    workflow = GAIAWorkflow(agent)
    
    # Run evaluation
    results = []
    correct = 0
    total = len(dataset)
    
    for i, task in enumerate(dataset):
        print(f"\nProgress: {i+1}/{total}")
        result = await workflow.run_single_task(task)
        results.append(result)
        
        if result.get("is_correct", False):
            correct += 1
            print(f"✅ Correct ({correct}/{i+1})")
        else:
            print(f"❌ Incorrect ({correct}/{i+1})")
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0.0
    
    evaluation_results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "model": str(agent.model.get_config().get("model_id", "unknown")),
        "detailed_results": results
    }
    
    # Save results if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print(f"\n=== GAIA Evaluation Summary ===")
    print(f"Model: {evaluation_results['model']}")
    print(f"Accuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"Average steps per task: {sum(r.get('num_steps', 0) for r in results) / len(results):.1f}")
    
    return evaluation_results


def create_gpt_oss_120b_agent() -> StrandsAgent:
    """Create a StrandsAgent with GPT-OSS 120B model."""
    # Setup tokenizer
    tokenizer_model = os.getenv("TOKENIZER_MODEL", "Qwen/Qwen3-4B")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    
    # Setup Together AI with GPT-OSS 120B
    together_api_key = os.getenv("TOGETHER_AI_API_KEY")
    if not together_api_key:
        raise ValueError("TOGETHER_AI_API_KEY must be set")
    
    openai_kwargs = {
        "api_key": together_api_key,
        "base_url": "https://api.together.xyz/v1"
    }
    
    # Create rollout engine and model
    rollout_engine = RolloutEngine(
        engine_name="openai", 
        tokenizer=tokenizer, 
        openai_kwargs=openai_kwargs
    )
    
    model = RLLMModel(
        rollout_engine=rollout_engine,
        model_id="openai/gpt-oss-120b",
        max_tokens=1000,
        temperature=0.1  # Lower temperature for more consistent evaluation
    )
    
    # Create agent with basic system prompt
    system_prompt = """You are a helpful AI assistant that can search the web and answer questions accurately. 
    When given a question, think step by step and provide a clear, direct answer."""
    
    agent = StrandsAgent(
        model=model,
        system_prompt=system_prompt
    )
    
    return agent


async def main():
    """Main evaluation script."""
    # Get evaluation parameters
    limit = int(os.getenv("GAIA_LIMIT", "10"))  # Default to 10 for testing
    output_file = os.getenv("GAIA_OUTPUT", "gaia_evaluation_results.json")
    
    print(f"Evaluating on {limit} GAIA tasks")
    print(f"Using model: openai/gpt-oss-120b")
    
    try:
        # Create agent
        agent = create_gpt_oss_120b_agent()
        
        # Run evaluation
        results = await evaluate_gaia(
            agent=agent,
            limit=limit,
            output_file=output_file
        )
        
        return results
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(main())