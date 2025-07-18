from rllm.agents.agent import Episode
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow, handle_termination, run_in_executor


class CritiqueWorkflow(Workflow):
    def __init__(
        self,
        solver_cls,
        critic_cls,
        env_cls,
        solver_args=None,
        critic_args=None,
        env_args=None,
        sampling_params=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Initialize mutable defaults
        solver_args = dict(solver_args) if solver_args is not None else {}
        critic_args = dict(critic_args) if critic_args is not None else {}
        env_args = dict(env_args) if env_args is not None else {}
        sampling_params = dict(sampling_params) if sampling_params is not None else {}

        self.solver = solver_cls(**solver_args)
        self.critic = critic_cls(**critic_args)
        self.env = env_cls(**env_args)
        self.sampling_params = sampling_params

        self.CRITIQUE_PROMPT = """### Question\n{question}\n\n### Answer\n{answer}\n\nYou are given a question and an answer to the question. Verify the answer step by step and generate a detailed critique that can help refine the answer. Be sure to clearly state if the provided answer is correct or incorrect."""
        self.OBSERVATION_PROMPT = """### Critique\n{critique}\n\nUsing the critique, refine your previous answer."""

    @handle_termination
    async def __call__(self, task: dict, uid: str, engine, **kwargs) -> Episode:
        """Execute a multi-step workflow"""

        # Reset environment using executor
        observation, info = await run_in_executor(engine.executor, self.env.reset, task)
        self.solver.reset(task=task)
        self.critic.reset()

        self.solver.update_from_env(observation, 0, False, info)
        response = await self.get_model_response(self.solver, uid, engine.rollout_engine, **self.sampling_params)
        action = self.solver.update_from_model(response)

        # Environment step using executor
        _, reward, done, info = await run_in_executor(engine.executor, self.env.step, action)

        critic_prompt = self.CRITIQUE_PROMPT.format(question=task.get("question", ""), answer=action.action)  # TODO: just the action?
        self.critic.update_from_env(critic_prompt, 0, False, info)
        critic_response = await self.get_model_response(engine.rollout_engine, self.critic.chat_completions, uid, **self.sampling_params)
        critique = self.critic.update_from_model(critic_response)

        next_obs = self.OBSERVATION_PROMPT.format(critique=critique.action)
        self.solver.update_from_env(next_obs, reward, done, info)
        response = await self.get_model_response(engine.rollout_engine, self.solver.chat_completions, uid, **self.sampling_params)
        action = self.solver.update_from_model(response)

        _, reward, done, info = await run_in_executor(engine.executor, self.env.step, action)
        self.solver.update_from_env(None, reward, done, info)
        self.critic.update_from_env(None, reward, done, info)  # give critic the solver's final reward

        raise TerminationEvent(TerminationReason.ENV_DONE)

    def assign_episode_correctness(self, episode: Episode) -> None:
        episode.is_correct = episode.trajectories["solver"].reward > 0
