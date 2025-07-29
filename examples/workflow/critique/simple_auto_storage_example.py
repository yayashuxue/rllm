#!/usr/bin/env python3
"""
Simple example showing auto-initialized episode storage.

This demonstrates how the AgentWorkflowEngine now automatically creates
an episode store, making it easier to track experiments.
"""

import asyncio
import os
from copy import deepcopy

from transformers import AutoTokenizer

from rllm.agents.critique_agent import CritiqueAgent
from rllm.agents.math_agent import MathAgent
from rllm.data.dataset_types import TestDataset
from rllm.data.utils import load_dataset
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout_engine import RolloutEngine
from rllm.environments.base.critique_env import CritiqueEnvironment
from rllm.rewards.reward_fn import math_reward_fn
from rllm.workflows.critique_workflow import CritiqueWorkflow


def process_math_fn(example, idx):
    question = example.pop("problem")
    instruction = "Let's think step by step, put your final answer within \\boxed{}."
    question = f"{question} {instruction}"
    answer = example.pop("answer")
    return {"ground_truth": answer, "question": question, "idx": idx, "data_source": "math"}


async def main():
    """Simple example with auto-initialized episode storage."""
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Load a small dataset for testing
    dataset = load_dataset(TestDataset.Math.AIME)

    


    tasks = [process_math_fn(deepcopy(example), idx) for idx, example in enumerate(list(dataset)[:3])]

    model_name = "Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    rollout_engine = RolloutEngine(
        engine_name="openai",
        tokenizer=tokenizer,
        openai_kwargs={
            "base_url": "http://localhost:30000/v1",
            "api_key": "None",
        },
        disable_thinking=False,
    )

    # 🎯 Simple setup - just specify where to store episodes!
    # No need to manually create episode store - it's auto-initialized
    engine = AgentWorkflowEngine(
        workflow_cls=CritiqueWorkflow,
        workflow_args={
            "solver_cls": MathAgent,
            "critic_cls": CritiqueAgent,
            "env_cls": CritiqueEnvironment,
            "solver_args": {"accumulate_thinking": False},
            "critic_args": {"accumulate_thinking": False},
            "env_args": {"reward_fn": math_reward_fn},
            "max_prompt_length": 8192,
            "max_response_length": 8192,
            "sampling_params": {"temperature": 0.6, "top_p": 0.95, "model": model_name},
        },
        rollout_engine=rollout_engine,
        n_parallel_tasks=10,
        retry_limit=1,
        default_db_path="simple_example_episodes.db",  # 🚀 Auto-creates SQLiteEpisodeStore!
    )

    print("🚀 Running simple example with auto-initialized episode storage...")
    print(f"📁 Episodes will be stored in: simple_example_episodes.db")

    # Execute tasks with workflow_id - episodes are automatically stored!
    workflow_id = "simple_critique_example"
    results = await engine.execute_tasks(tasks, workflow_id=workflow_id)

    print(f"\n✅ Completed {len(results)} episodes")
    
    # 📊 Check what was stored (episode store is accessible via engine.episode_store)
    stats = engine.episode_store.get_statistics(workflow_id)
    print(f"📈 Results for '{workflow_id}':")
    print(f"   Total episodes: {stats['total_episodes']}")
    print(f"   Correct episodes: {stats['correct_episodes']}")
    print(f"   Accuracy: {stats['accuracy']:.1%}")

    # Clean up
    engine.shutdown()
    print("\n🎉 Example completed! Check 'simple_example_episodes.db' for stored episodes.")


if __name__ == "__main__":
    asyncio.run(main()) 