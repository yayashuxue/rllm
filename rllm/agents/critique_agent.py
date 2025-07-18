import copy
from typing import Any

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory


class CritiqueAgent(BaseAgent):
    """A generic agent that takes an observation of type str."""

    def __init__(self, accumulate_thinking=True):
        self._trajectory = Trajectory()
        self.messages = []
        self.accumulate_thinking = accumulate_thinking

    def update_from_env(self, observation: str, reward: float, done: bool, info: dict, **kwargs):
        """
        Updates the agent's internal state after an environment step.
        """

        # If there are previous steps, update the last step's outcome
        if self.trajectory.steps:
            cur_step = self.get_current_state()
            cur_step.reward = reward
            cur_step.done = done
            cur_step.info = info

        if done:
            return

        self.messages.append({"role": "user", "content": observation})

        new_step = Step(observation=observation)
        self._trajectory.steps.append(new_step)

    def update_from_model(self, response: Any, **kwargs):
        """
        Updates the agent's internal state based on the model's response.
        """

        # Update the latest step
        self.messages.append({"role": "assistant", "content": response})

        cur_step = self.get_current_state()
        cur_step.chat_completions = self.chat_completions
        cur_step.model_response = response

        if response.count("</think>") == 1:
            thought, sep, action = response.partition("</think>")
            thought = thought + sep
            action = Action(action.strip())
        else:
            thought = None
            action = Action(response.strip())

        cur_step.thought = thought
        cur_step.action = action

        return action

    def reset(self, task: Any = None):
        """
        Resets the agent's internal state for a new episode.
        """
        self._trajectory = Trajectory(task=task)
        self.messages = []

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Return conversation history for model interaction."""
        # remove thinking from assistant messages if not accumulate_thinking except the last one
        messages = copy.deepcopy(self.messages)
        if not self.accumulate_thinking:
            for msg in messages[:-1]:
                if msg["role"] == "assistant":
                    _, sep, after = msg["content"].partition("</think>")
                    if sep:
                        msg["content"] = after
        return messages

    @property
    def trajectory(self) -> Trajectory:
        """Returns the trajectory object."""
        return self._trajectory

    def get_current_state(self) -> Step:
        """Returns the current step/state of the agent."""
        assert self._trajectory.steps, "Trajectory should not be empty when get_current_state is called."
        return self._trajectory.steps[-1]
