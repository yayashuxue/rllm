import warnings
from collections import deque
from typing import Any

from rllm.environments.base.multi_turn_env import MultiTurnEnvironment
from rllm.rewards.reward_fn import RewardFunction, zero_reward


class SolveVerifyEnvironment(MultiTurnEnvironment):
    """
    An environment to support the verification workflows.
    Supports multi-turn iteractions between a solver and verifier.
    **It returns empty observation.**
    """

    def __init__(self, task: dict | None = None, reward_fn: RewardFunction | None = None, max_consecutive_correct_verifications: int = 3, **kwargs):
        """
        Initialize the single turn environment.

        Args:
            task: Dictionary containing the task information, including at least a "question" field
        """
        super().__init__(task=task, max_turns=1000, **kwargs)  # max turns will be thrown by the workflow
        if reward_fn is None:
            warnings.warn("No reward function provided, using zero reward", stacklevel=2)
        self.reward_fn = reward_fn or zero_reward
        self.deque = deque(maxlen=max_consecutive_correct_verifications)

    def step(self, action: Any) -> tuple[dict, float, bool, dict]:
        raise NotImplementedError("Please use step_solver or step_verifier")

    def step_solver(self, action: Any) -> tuple[dict, float, bool, dict]:
        """
        Step the solver.
        """
        # Calculate reward for the current turn using the reward_fn
        assert self.task is not None, "Task is not set"
        reward = self.reward_fn(task_info=self.task, action=action).reward

        # Store the (action, reward) in history
        self.history.append((action, reward))

        # Increment turn counter
        self.current_turn += 1

        return {}, reward, False, self.task

    def step_verifier(self, action: Any) -> tuple[dict, float, bool, dict]:
        """
        Step the verifier.
        """
        assert self.history and len(self.history) > 0, "Verifier must be called after the Solver"

        # ground truth is based on the last solver's previous reward
        # any positive reward is considered correct
        gt = str(int(self.history[-1][1] > 0))

        # use the math reward function to check the correctness of the verification agianst gt
        reward = self.reward_fn(task_info={"ground_truth": gt}, action=action).reward

        # Backtrack to get pred
        if (gt == "1" and reward > 0) or (gt == "0" and reward <= 0):
            pred = int(gt)
        else:
            pred = 1 - int(gt)

        # update the deque
        self.deque.append(pred)

        # if the last max_consecutive_correct_verifications verifications are positive (i.e., verifier predicts correct), return done
        if len(self.deque) == self.deque.maxlen and all(pred == 1 for pred in self.deque):
            return {}, reward, True, self.task

        return {}, reward, False, self.task

    def get_reward_and_next_obs(self, task: dict, action: Any) -> tuple[float, dict]:
        pass

    @staticmethod
    def from_dict(env_args: dict) -> "SolveVerifyEnvironment":
        return SolveVerifyEnvironment(**env_args)
