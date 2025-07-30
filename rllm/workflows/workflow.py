import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from functools import partial
from typing import Any

import numpy as np

from rllm.agents.agent import BaseAgent, Episode, Trajectory
from rllm.engine.rollout_engine import RolloutEngine
from rllm.environments.base.base_env import BaseEnv


def handle_termination(func: Callable):
    """Decorator that handles termination errors and breaks the loop"""

    async def wrapper(self, task: dict, uid: str, **kwargs):
        try:
            coro = func(self, task, uid, **kwargs)
            return await asyncio.wait_for(coro, timeout=self.timeout)
        except asyncio.TimeoutError:
            self.barrier.mark_terminated(uid)
            return self.postprocess_episode(self.collect_trajectories(), TerminationReason.TIMEOUT)
        except TerminationEvent as e:
            self.barrier.mark_terminated(uid)
            return self.postprocess_episode(self.collect_trajectories(), e.reason)

    return wrapper


class TerminationReason(Enum):
    MAX_PROMPT_LENGTH_EXCEEDED = "max_prompt_length_exceeded"
    MAX_RESPONSE_LENGTH_EXCEEDED = "max_response_length_exceeded"
    ENV_DONE = "env_done"
    MAX_TURNS_EXCEEDED = "max_turns_exceeded"
    TIMEOUT = "timeout"
    NOT_ENOUGH_PEERS = "not_enough_peers"


class TerminationEvent(Exception):
    def __init__(self, reason: TerminationReason = None):
        super().__init__(f"Terminated: {reason}")
        self.reason = reason


class Workflow(ABC):
    def __init__(self, rollout_engine: RolloutEngine, executor: ThreadPoolExecutor, barrier, max_prompt_length=4096, max_response_length=8192, accumulate_response_length=True, timeout=None, gamma=0.0, reward_bonus_coeff=0.0, **kwargs):
        self.rollout_engine = rollout_engine
        self.executor = executor
        self.barrier = barrier
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.accumulate_response_length = accumulate_response_length
        self.timeout = timeout or int(1e9)
        self.gamma = gamma
        self.reward_bonus_coeff = reward_bonus_coeff

        self.agent_registry: dict[BaseAgent, dict[str, Any]] = {}

    @abstractmethod
    @handle_termination
    async def __call__(self, task: dict, uid: str, **kwargs) -> Episode:
        """Execute the workflow on a single task"""
        pass

    def register_agent(self, agent: BaseAgent, rollout_engine: RolloutEngine | None = None) -> None:
        """Register an agent and its rollout engine"""
        self.agent_registry[agent] = {
            "rollout_engine": rollout_engine or self.rollout_engine,
            "accumulated_response_length": 0,
        }

    def collect_trajectories(self) -> Episode:
        """Collect the trajectories from the workflow"""

        episode = Episode()

        for attr_name in dir(self):
            # Skip private attributes and methods
            if attr_name.startswith("_"):
                continue

            attr_value = getattr(self, attr_name)

            # Check if attribute is a BaseAgent instance
            if isinstance(attr_value, BaseAgent) and hasattr(attr_value, "trajectory"):
                episode.trajectories.append((attr_name, attr_value.trajectory))

        assert len(episode.trajectories) > 0, "No trajectories found in the workflow"

        return episode

    def compute_trajectory_reward(self, agent_name: str, trajectory: Trajectory) -> None:
        """
        Compute the trajectory-level reward.
        Default: sum the step rewards
        """
        trajectory.reward = np.sum([d.reward for d in trajectory.steps])

    def adjust_step_rewards(self, agent_name: str, trajectory: Trajectory) -> None:
        """
        Adjust the step-level rewards.
        Default: Reward shaping and discounting
        self.reward_bonus_coeff and self.gamma are 0.0, so no adjustments are made by default.
        """
        # reward shaping
        # s[i].reward = s[i].reward + bonus * (s[i].reward - s[i-1].reward) for i > 0
        if self.reward_bonus_coeff > 0.0:
            raw_rewards = [step.reward for step in trajectory.steps]
            for i in range(1, len(trajectory.steps)):
                trajectory.steps[i].reward += self.reward_bonus_coeff * (raw_rewards[i] - raw_rewards[i - 1])

        # Compute Monte Carlo returns (backward iteration)
        # G_t = R_{t+1} + γ * R_{t+2} + γ² * R_{t+3} + ... + γ^{T-t-1} * R_T
        if self.gamma > 0.0:
            G = 0.0
            for step in reversed(trajectory.steps):
                G = step.reward + self.gamma * G
                step.reward = G  # Replace the reward with MC return

    def assign_episode_correctness(self, episode: Episode) -> None:
        """
        Assign an episode-level correctness flag.
        Default: True if the sum of the trajectory rewards is strictly positive.
        """
        total_reward = 0
        for agent_name, trajectory in episode.trajectories:
            total_reward += trajectory.reward
        episode.is_correct = total_reward > 0

    def postprocess_episode(self, episode: Episode, termination_reason: TerminationReason = None) -> Episode:
        """Collect and process the trajectories"""
        assert episode is not None, "Remember to call collect_trajectories() before postprocessing the episode"

        # 1. assign a task id and task
        episode.id = self.uid
        episode.task = self.task

        for agent_name, trajectory in episode.trajectories:
            # depending on the terminaiton reason, there should be a trajectry with an additional step with empty chat_completions
            # i.e., if it's thrown between agent.update_from_env() and agent.update_from_model()
            # e.g., TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED
            if trajectory.steps and not trajectory.steps[-1].chat_completions:
                trajectory.steps.pop()

            # 2. compute trajectory-level rewards
            self.compute_trajectory_reward(agent_name, trajectory)

            # 3. adjust the step level rewards (e.g., reward shaping or discounting)
            if len(trajectory.steps) > 1:
                self.adjust_step_rewards(agent_name, trajectory)

        # 4. assign an episode-level correctness flag
        self.assign_episode_correctness(episode)

        # 5. assign a termination reason
        episode.termination_reason = termination_reason

        return episode

    def reset(self, task: dict | None = None, uid: str | None = None) -> tuple[Any, dict]:
        """Reset all agents and environments for reuse"""

        # set the uid and task
        self.uid = uid
        self.task = task

        # reset the agents in the registry
        for agent in self.agent_registry:
            agent.reset()
            agent.trajectory.task = task
            self.agent_registry[agent]["accumulated_response_length"] = 0  # reset response length counter

        # reset environments (look for class attributes that are BaseEnv subclasses
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, BaseEnv) and hasattr(attr_value, "reset"):
                return attr_value.reset(task=task)

        raise ValueError("No environment found in the workflow")

    def is_multithread_safe(self) -> bool:
        """Check if the workflow is multithread safe"""
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, BaseEnv) and not attr_value.is_multithread_safe():
                return False
        return True

    async def run_in_executor(self, fn, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, partial(fn, *args, **kwargs))

    async def get_model_response(self, agent: BaseAgent, messages: list[dict] | None = None, **kwargs) -> str:
        """Get the model response for the given messages"""

        if agent not in self.agent_registry:
            raise ValueError(f"Agent {agent} is not registered with the workflow. Please call register_agent() during initialization.")

        rollout_engine = self.agent_registry[agent]["rollout_engine"]
        # TODO: add post init logic to ensure at least one agent is using the workflow engine's rollout engine

        messages = messages or agent.chat_completions

        assert messages[-1]["role"] != "assistant", "Prefilling the assistant message is not supported"

        # We check if the prompt length exceeds the max prompt length, and if so, we do not generate a response
        if self.rollout_engine.chat_parser is not None:
            prompt = rollout_engine.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True)
            prompt_length = len(rollout_engine.tokenizer.encode(prompt))
            if prompt_length > self.max_prompt_length:
                print(f"Prompt length {prompt_length} exceeds max prompt length {self.max_prompt_length} for rollout {self.uid}")
                raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        # We deal with response length in one of two ways:
        # 1. If accumulate_response_length==True, then we keep a running counter of the accumulated response length seperately for each agent in the workflow
        #    (note: response = observations (except the first) + assistant generations) and generate with max_tokens=self.max_response_length - self.accumulated_response_length.
        # 2. if accumulate_response_length=False, then each response is generated with max_tokens=self.max_response_length.
        if self.accumulate_response_length and self.rollout_engine.chat_parser is not None:
            accumulated_response_length = self.agent_registry[agent]["accumulated_response_length"]

            # count the number of tokens in the last observation (i.e., the messages since the last assistant message)
            last_assistant_idx = next((i for i, m in reversed(list(enumerate(messages))) if m["role"] == "assistant"), None)
            if last_assistant_idx is not None:  # must be the initial prompt
                last_observation = messages[last_assistant_idx + 1 :]
                observation_length = len(rollout_engine.tokenizer.encode(rollout_engine.chat_parser.parse(last_observation, add_generation_prompt=True, is_first_msg=False)))
                accumulated_response_length += observation_length

            max_tokens = self.max_response_length - accumulated_response_length
            if max_tokens <= 0:
                print(f"Rollout {self.uid} reached max response length")
                raise TerminationEvent(TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)

            self.agent_registry[agent]["accumulated_response_length"] = accumulated_response_length
        else:
            max_tokens = self.max_response_length

        response = await rollout_engine.get_model_response(messages, application_id=self.uid, max_tokens=max_tokens, **kwargs)

        # TODO: throw TerminationEvent.MAX_RESPONSE_LENGTH_EXCEEDED based on response (requires reconfiguring the rollout engine and router)
        # ideally the rollout_engine returns the response dict (e.g., CompletionOutput or ChatCompletionOutput) not a string

        if self.accumulate_response_length and self.rollout_engine.chat_parser is not None:
            # TODO: technically, we're undercounting here a bit (e.g., by 2 per turn for Qwen3) because the chat
            # parser will add the eot token(s) to the response, which is not returned by the engine
            response_length = len(rollout_engine.tokenizer.encode(response))
            self.agent_registry[agent]["accumulated_response_length"] += response_length
        return response
