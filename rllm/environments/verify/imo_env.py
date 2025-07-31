from typing import Any

from rllm.environments.base.multi_turn_env import MultiTurnEnvironment
from rllm.rewards.reward_fn import zero_reward


class IMOEnvironment(MultiTurnEnvironment):
    """
    An environment to support the IMO workflows.
    Supports multi-turn iteractions between a solver and verifier.
    **It returns empty observation.**
    """

    def __init__(self, task: dict | None = None, **kwargs):
        """
        Initialize the single turn environment.

        Args:
            task: Dictionary containing the task information, including at least a "question" field
        """
        super().__init__(task=task, max_turns=1000, **kwargs)  # max turns will be thrown by the workflow
        self.reward_fn = zero_reward

    def step(self, action: Any) -> tuple[dict, float, bool, dict]:
        raise NotImplementedError("Please use step_solver or step_verifier")

    def step_solver(self, action: Any) -> tuple[dict, float, bool, dict]:
        """
        Step the solver.
        """
        reward = self.reward_fn(task_info=self.task, action=action).reward
        return {}, reward, False, self.task

    def step_verifier(self, action: Any) -> tuple[dict, float, bool, dict]:
        """
        Step the verifier.
        """
        reward = self.reward_fn(task_info=self.task, action=action).reward
        return {}, reward, False, self.task

    def get_reward_and_next_obs(self, task: dict, action: Any) -> tuple[float, dict]:
        pass

    @staticmethod
    def from_dict(env_args: dict) -> "IMOEnvironment":
        return IMOEnvironment(**env_args)
