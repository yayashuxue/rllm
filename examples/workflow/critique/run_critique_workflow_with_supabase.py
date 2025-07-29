#!/usr/bin/env python3
"""
Example using Supabase for episode storage.

This demonstrates how to use SupabaseEpisodeStore to store episodes in the cloud
instead of a local SQLite database.

Prerequisites:
1. Create a Supabase project at https://supabase.com
2. Run the SQL migration: rllm/db/supabase_migration.sql
3. Install supabase: pip install supabase
4. Set environment variables or provide credentials directly
"""

import asyncio
import json
import os
from copy import deepcopy

from transformers import AutoTokenizer

from rllm.agents.critique_agent import CritiqueAgent
from rllm.agents.math_agent import MathAgent
from rllm.data.dataset_types import TestDataset
from rllm.data.utils import load_dataset
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout_engine import RolloutEngine
from rllm.db.episode_store import SupabaseEpisodeStore
from rllm.environments.base.critique_env import CritiqueEnvironment
from rllm.rewards.reward_fn import math_reward_fn
from rllm.workflows.critique_workflow import CritiqueWorkflow


def load_data(n=1, dataset_enum=None):
    """Load data using the new Dataset interface."""
    dataset = load_dataset(dataset_enum)
    data = []
    for idx, example in enumerate(dataset):
        processed = process_math_fn(example, idx)
        for i in range(n):
            data.append(deepcopy(processed))
    return data


def process_math_fn(example, idx):
    question = example.pop("problem")
    instruction = "Let's think step by step, put your final answer within \\boxed{}."
    question = f"{question} {instruction}"
    answer = example.pop("answer")

    task = {"ground_truth": answer, "question": question, "idx": idx, "data_source": "math"}
    return task


def evaluate_results(results):
    from collections import defaultdict

    # Create a map to store correct answers per problem
    problem_correct_map = defaultdict(int)
    problem_total_map = defaultdict(int)

    # Count correct answers for each problem
    for episode in results:
        problem = episode.task["question"]
        is_correct = episode.is_correct

        problem_correct_map[problem] += is_correct
        problem_total_map[problem] += 1

    # Calculate pass@1 and pass@k
    k = max(problem_total_map.values())
    total_problems = len(problem_correct_map)
    pass_at_1 = sum(problem_correct_map.values()) / sum(problem_total_map.values())
    pass_at_k = sum(1 for problem, correct in problem_correct_map.items() if correct > 0) / total_problems

    print("Total unique problems:", total_problems)
    print("Average Pass@1 Accuracy:", pass_at_1)
    print(f"Average Pass@{k} Accuracy:", pass_at_k)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Check for Supabase credentials
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("❌ Supabase credentials required!")
        print("   Set environment variables:")
        print("   export SUPABASE_URL='https://your-project.supabase.co'")
        print("   export SUPABASE_KEY='your-anon-key'")
        print()
        print("   Or provide them directly in the SupabaseEpisodeStore constructor")
        print("   Alternatively, use run_critique_workflow_with_storage.py for SQLite")
        exit(1)

    # Create the environment (no batch_size parameter)
    n_parallel_tasks = 30
    workflow_id = "critique_math_supabase_001"  # Specify a workflow ID

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

    print("🚀 Initializing Supabase episode store...")
    
    try:
        # Option 1: Use environment variables (recommended)
        episode_store = SupabaseEpisodeStore()
        
        # Option 2: Provide credentials directly
        # episode_store = SupabaseEpisodeStore(
        #     supabase_url="https://your-project.supabase.co",
        #     supabase_key="your-anon-key"
        # )
        
        print("✅ Connected to Supabase successfully!")
        
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")
        print("   Please check your credentials and network connection")
        print("   Make sure you've run the migration: rllm/db/supabase_migration.sql")
        exit(1)

    engine = AgentWorkflowEngine(
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
            "sampling_params": {"temperature": 0.6, "top_p": 0.95, "model": model_name},
        },
        rollout_engine=rollout_engine,
        config=None,
        n_parallel_tasks=n_parallel_tasks,
        retry_limit=1,
        episode_store=episode_store,  # Use Supabase store
    )

    # tasks = load_data(n=1, dataset_enum=TestDataset.Math.AIME)
    from rllm.data.dataset import DatasetRegistry
    tasks = DatasetRegistry.load_dataset("aime2024", "test")[:1]

    print(f"📊 Executing {len(tasks)} tasks with Supabase storage...")

    # Execute tasks with workflow_id - episodes will be automatically stored in Supabase!
    results = asyncio.run(engine.execute_tasks(tasks, workflow_id=workflow_id))
    evaluate_results(results)

    # Save results to JSON as before
    os.makedirs("logs", exist_ok=True)
    with open("logs/critique_workflow_supabase.json", "w") as f:
        json.dump([episode.to_dict() for episode in results], f, indent=4)

    # Demonstrate Supabase episode store functionality
    print(f"\n--- Supabase Episode Store Statistics for workflow '{workflow_id}' ---")
    try:
        stats = engine.episode_store.get_statistics(workflow_id)
        print(f"☁️  Total episodes stored in Supabase: {stats['total_episodes']}")
        print(f"✅ Correct episodes: {stats['correct_episodes']}")
        print(f"📈 Accuracy: {stats['accuracy']:.2%}")

        # Retrieve episodes from Supabase
        stored_episodes = engine.episode_store.get_episodes(workflow_id, limit=5)
        print(f"\n📥 Retrieved {len(stored_episodes)} episodes from Supabase (limited to 5)")
        for i, episode in enumerate(stored_episodes):
            print(f"  Episode {i+1}: ID={episode.id}, Correct={episode.is_correct}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not retrieve statistics from Supabase: {e}")

    # Clean up
    engine.shutdown()

    print(f"\n🎉 Episodes stored in Supabase with workflow_id: {workflow_id}")
    print("   You can now:")
    print("   • View episodes in your Supabase dashboard")
    print("   • Query episodes from other scripts using the same workflow_id")
    print("   • Access episodes from anywhere with internet connection")
    print("   • Use Supabase's real-time features for live monitoring") 