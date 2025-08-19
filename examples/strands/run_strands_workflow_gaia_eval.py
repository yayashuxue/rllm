"""Run GAIA evaluation using the Workflow engine + Strands agent."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, List

from transformers import AutoTokenizer

from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout_engine import RolloutEngine

from gaia_workflow import StrandsGaiaWorkflow

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

def load_gaia_dataset(limit: int | None = None) -> List[dict[str, Any]]:
    try:
        from rllm.data.utils import load_dataset
        from rllm.data.dataset_types import TestDataset
        data = load_dataset(TestDataset.Web.GAIA)
        dataset = list(data) if not isinstance(data, list) else data
    except Exception:
        # Fallback to local json
        import rllm
        rllm_path = os.path.dirname(os.path.dirname(rllm.__file__))
        data_path = os.path.join(rllm_path, "rllm/data/train/web/gaia.json")
        with open(data_path, "r") as f:
            dataset = json.load(f)
    if limit:
        dataset = dataset[:limit]
    return dataset


async def main():
    print("🔎 GAIA Evaluation (Workflow Engine + Strands Agent)")

    # Resolve provider/model like create_browser_agent
    together_api_key = os.getenv("TOGETHER_AI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if together_api_key:
        api_key = together_api_key
        base_url = os.getenv("OPENAI_BASE_URL") or "https://api.together.xyz/v1"
        model_name = os.getenv("TOGETHER_AI_MODEL_NAME", os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct-Turbo"))
    elif openai_api_key:
        api_key = openai_api_key
        base_url = os.getenv("OPENAI_BASE_URL")
        model_name = os.getenv("MODEL_NAME", "gpt-4o")
    else:
        raise ValueError("API key required: set TOGETHER_AI_API_KEY or OPENAI_API_KEY in env/.env")

    limit = int(os.getenv("GAIA_LIMIT", "2"))

    print(f"Model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(os.getenv("TOKENIZER_MODEL", "Qwen/Qwen3-4B"))

    openai_kwargs = {"api_key": api_key} if api_key else {}
    if base_url:
        openai_kwargs["base_url"] = base_url

    rollout_engine = RolloutEngine(
        engine_name="openai",
        tokenizer=tokenizer,
        openai_kwargs=openai_kwargs,
    )

    engine = AgentWorkflowEngine(
        workflow_cls=StrandsGaiaWorkflow,
        workflow_args={
            "model_id": model_name,
            "model_params": {"temperature": 0.2, "max_tokens": 1000},
        },
        rollout_engine=rollout_engine,
        config=None,
        n_parallel_tasks=1,
    )

    dataset = load_gaia_dataset(limit=limit)
    print(f"Loaded {len(dataset)} GAIA tasks")

    out_path = os.getenv("GAIA_OUTPUT", "gaia_workflow_results.json")
    resume = os.getenv("GAIA_RESUME", "0") == "1"
    episodes_accum: list[dict] = []
    done_ids: set[str] = set()

    # Load previous results if resume flag is set
    if resume and os.path.exists(out_path):
        try:
            with open(out_path, "r") as f:
                prev = json.load(f)
            episodes_accum = list(prev.get("episodes", []))
            for ep in episodes_accum:
                task_obj = ep.get("task") or {}
                tid = task_obj.get("task_id")
                if tid:
                    done_ids.add(tid)
            print(f"Resuming: found {len(episodes_accum)} previously saved episodes; will skip {len(done_ids)} tasks")
        except Exception as e:
            print(f"Resume disabled due to load error: {e}")
            episodes_accum = []
            done_ids = set()

    # Filter dataset if resuming
    if done_ids:
        dataset = [t for t in dataset if t.get("task_id") not in done_ids]
        print(f"Remaining tasks: {len(dataset)}")

    # Helper to save incrementally
    def save_partial():
        correct = sum(1 for ep in episodes_accum if ep.get("is_correct", False))
        total = len(episodes_accum)
        out = {
            "accuracy": (correct / total) if total else 0.0,
            "correct": correct,
            "total": total,
            "model": model_name,
            "episodes": episodes_accum,
        }
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, default=str)

    # Process tasks sequentially to save after each episode
    print("Generating trajectories (stream save enabled)")
    for idx, task in enumerate(dataset, start=1):
        episodes = await engine.execute_tasks([task])
        ep = episodes[0]
        ep_dict = getattr(ep, "to_dict", lambda: {})() if hasattr(ep, "to_dict") else {
            "id": getattr(ep, "id", None),
            "task": getattr(ep, "task", None),
            "is_correct": getattr(ep, "is_correct", False),
        }
        episodes_accum.append(ep_dict)
        save_partial()
        correct_so_far = sum(1 for e in episodes_accum if e.get("is_correct", False))
        print(f"[{idx}/{len(dataset)}] saved. Running accuracy: {correct_so_far}/{len(episodes_accum)} ({(correct_so_far/len(episodes_accum)*100):.1f}%)")

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())


