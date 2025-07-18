import asyncio
import json
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
    for trajectories in results:
        trajectory = trajectories[0]
        problem = trajectory.steps[0].observation

        is_correct = 1 if trajectory.reward > 0 else 0

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
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Create the environment (no batch_size parameter)
    n_parallel_tasks = 256

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
            "enforce_max_prompt_length": True,
            "accumulate_response_length": True,
        },
        rollout_engine=rollout_engine,
        config=None,
        n_parallel_tasks=n_parallel_tasks,
    )

    engine.init_workflows()

    tasks = load_data(n=16, dataset_enum=TestDataset.Math.AIME)

    results = asyncio.run(engine.execute_tasks(tasks))
    evaluate_results(results)

    results = [[traj.to_dict() for traj in result] for result in results]
    with open("logs/test.json", "w") as f:
        json.dump(results, f, indent=4)
