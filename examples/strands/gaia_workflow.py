"""
GAIA evaluation workflow using StrandsAgent and AgentWorkflowEngine.

Keeps logic simple: one agent, tool-aware prompt, model tool-calls enabled via
default tool specs set by StrandsAgent. No custom interactive loop.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from rllm.engine.rollout_engine import RolloutEngine
from rllm.workflows.workflow import Workflow, handle_termination
from rllm.agents.agent import Episode
from rllm.integrations.strands import RLLMModel, StrandsAgent

# Ensure local imports work when executed as a module or script
import os as _os, sys as _sys
_sys.path.append(_os.path.dirname(__file__))
# Reuse the agent factory to ensure identical prompt and tools
from agent_factory import create_browser_agent


class StrandsGaiaWorkflow(Workflow):
    """Minimal GAIA workflow backed by a single StrandsAgent."""

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        model_id: Optional[str] = None,
        model_params: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(rollout_engine=rollout_engine, **kwargs)

        # Build a StrandsAgent with the same prompt/tools as other examples
        # Note: create_browser_agent already wires system prompt and tools
        self.agent, self.browser, self.executor = create_browser_agent()

        # Ensure the agent uses this workflow's rollout engine and update config
        self.agent.model.rollout_engine = self.rollout_engine
        if model_params:
            self.agent.model.update_config(**model_params)
        if model_id:
            self.agent.model.update_config(model_id=model_id)

    # Override base reset to avoid requiring a BaseEnv
    def reset(self, task: dict | None = None, uid: str | None = None) -> tuple[Any, dict] | None:
        self.uid = uid
        self.task = task
        # Reset agent state and bind task on trajectory
        try:
            self.agent.reset()
        except Exception:
            pass
        # Hard clear Strands Agent message buffer to avoid cross-task bleed
        try:
            if hasattr(self.agent, "messages") and isinstance(self.agent.messages, list):
                self.agent.messages.clear()
        except Exception:
            pass
        self.agent.trajectory.task = task
        return None

        # StrandsAgent is not a BaseAgent; skip registry-based collection

    def _extract_text(self, result: Any) -> str:
        if hasattr(result, "message") and result.message:
            content = getattr(result.message, "content", None)
            if isinstance(content, list):
                text = "".join(str(cb.get("text", "")) for cb in content if isinstance(cb, dict))
                return text.strip()
            if content is not None:
                return str(content).strip()
        return str(result).strip() if result is not None else ""

    def _get_question_and_answer(self, task: dict[str, Any]) -> tuple[str, str]:
        # Accept both GAIA formats
        question = task.get("Question") or task.get("problem") or task.get("question") or ""
        ground = task.get("Final answer") or task.get("tests") or task.get("ground_truth") or ""
        return str(question), str(ground)

    @handle_termination
    async def __call__(self, task: dict, uid: str, **kwargs) -> Any:
        # Reset agent for a fresh trajectory
        self.reset(task=task, uid=uid)
        self.agent.reset_trajectory(task=task)

        # Prepare episode container early to ensure we always return a valid Episode
        episode = Episode()
        episode.trajectories.append(("agent", self.agent.trajectory))

        question, ground_truth = self._get_question_and_answer(task)

        # Multi-turn loop driven by MAX_STEPS/GAIA_MAX_STEPS; Strands manages chat/tooling
        if "MAX_STEPS" not in _os.environ:
            _os.environ["MAX_STEPS"] = _os.getenv("GAIA_MAX_STEPS", "10")
        max_steps = int(_os.getenv("GAIA_MAX_STEPS", _os.getenv("MAX_STEPS", "10")))

        # Let Strands manage chat history and tool calls; we just allow multiple invokes
        latest_text = ""
        for step in range(1, max_steps + 1):
            user_prompt = question if step == 1 else "Continue with next Thought and Action."
            result = await self.agent.invoke_async(user_prompt)
            latest_text = self._extract_text(result)
            # Optional: early stop if model produced a final answer line
            if "final_answer" in (latest_text or "").lower():
                break

        pred = (latest_text or "").lower().strip()
        gold = str(ground_truth).lower().strip()
        is_correct = bool(pred) and bool(gold) and (pred == gold)
        self.agent.update_step_reward(1.0 if is_correct else 0.0)
        episode.is_correct = is_correct
        # Ensure trajectory task matches the current task before saving
        try:
            self.agent.trajectory.task = task
        except Exception:
            pass
        return self.postprocess_episode(episode)


