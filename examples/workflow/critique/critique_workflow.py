import argparse
import asyncio
import json
import os
from copy import deepcopy

from transformers import AutoTokenizer

from rllm.agents.critique_agent import CritiqueAgent
from rllm.agents.math_agent import MathAgent
from rllm.api import serve
from rllm.data.dataset import DatasetRegistry
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


def create_engine(n_parallel_tasks=30):
    """Create and return the critique workflow engine."""
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
        },
        rollout_engine=rollout_engine,
        config=None,
        n_parallel_tasks=n_parallel_tasks,
        retry_limit=1,
    )
    
    return engine


def run_evaluation():
    """Run the workflow evaluation on the test dataset."""
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    engine = create_engine(n_parallel_tasks=30)
    
    # Load test tasks
    tasks = DatasetRegistry.load_dataset("aime2024", "test")
    
    # Run evaluation
    results = asyncio.run(engine.execute_tasks(tasks))
    evaluate_results(results)
    
    # Save results
    os.makedirs("logs", exist_ok=True)
    with open("logs/critique_workflow.json", "w") as f:
        json.dump([episode.to_dict() for episode in results], f, indent=4)


def run_server(port=8001):
    """Serve the critique workflow via HTTP."""
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    engine = create_engine(n_parallel_tasks=10)  # Fewer tasks for serving
    
    # Serve the workflow
    serve(engine, workflow_name="critique", port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Critique Workflow - Run evaluation or serve via HTTP")
    parser.add_argument(
        "mode", 
        choices=["eval", "serve"], 
        help="Mode: 'eval' to run evaluation, 'serve' to start HTTP server"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8001, 
        help="Port for HTTP server (default: 8001)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "eval":
        run_evaluation()
    elif args.mode == "serve":
        run_server(port=args.port)
