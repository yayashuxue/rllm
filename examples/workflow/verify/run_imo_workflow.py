import asyncio
import json
from copy import deepcopy

from rllm.agents.math_agent import MathAgent
from rllm.data import DatasetRegistry
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout_engine import RolloutEngine
from rllm.environments.verify.imo_env import IMOEnvironment
from rllm.workflows.imo_workflow import IMOWorkflow


def load_data(n=1):
    """Load data using the new Dataset interface."""
    dataset = DatasetRegistry.load_dataset("imo", "test")
    data = []
    for example in dataset:
        for i in range(n):
            data.append(deepcopy(example))
    return data


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
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Create the environment (no batch_size parameter)
    n_parallel_tasks = 1

    model_name = "o4-mini-2025-04-16"

    rollout_engine = RolloutEngine(
        engine_name="openai",
        openai_kwargs={
            "base_url": "https://api.openai.com/v1",
            "api_key": os.getenv("OPENAI_API_KEY", ""),
        },
        api_retries=1,
    )

    engine = AgentWorkflowEngine(
        workflow_cls=IMOWorkflow,
        workflow_args={
            "solver_cls": MathAgent,
            "verifier_cls": MathAgent,
            "env_cls": IMOEnvironment,
            "max_prompt_length": 16384,
            "max_response_length": 16384,
            "sampling_params": {"model": model_name, "reasoning_effort": "high"},
        },
        rollout_engine=rollout_engine,
        config=None,
        n_parallel_tasks=n_parallel_tasks,
        retry_limit=1,
    )

    tasks = load_data(n=4)
    task_ids = [str(x["id"]) for x in tasks]

    results = asyncio.run(engine.execute_tasks(tasks, task_ids))
    evaluate_results(results)

    with open("logs/imo_workflow.json", "w") as f:
        json.dump([episode.to_dict() for episode in results], f, indent=4)
