from rllm.agents.agent import Episode
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow, handle_termination, run_in_executor


class MultiTurnWorkflow(Workflow):
    def __init__(
        self,
        agent_cls,
        env_cls,
        agent_args=None,
        env_args=None,
        max_steps=5,
        sampling_params=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Initialize mutable defaults
        agent_args = dict(agent_args) if agent_args is not None else {}
        env_args = dict(env_args) if env_args is not None else {}
        sampling_params = dict(sampling_params) if sampling_params is not None else {}

        self.agent = agent_cls(**agent_args)
        self.env = env_cls(**env_args)
        self.max_steps = max_steps
        self.sampling_params = sampling_params

    @handle_termination
    async def __call__(self, task: dict, uid: str, engine, **kwargs) -> Episode:
        """Execute a multi-step workflow"""

        # Reset environment using executor
        observation, info = await run_in_executor(engine.executor, self.env.reset, task)
        self.agent.reset(task=task)
        self.agent.update_from_env(observation, 0, False, info)

        for step in range(1, self.max_steps + 1):
            prompt = self.agent.chat_completions
            response = await self.get_model_response(engine.rollout_engine, prompt, uid, **self.sampling_params)
            action = self.agent.update_from_model(response)

            # Environment step using executor
            next_obs, reward, done, info = await run_in_executor(engine.executor, self.env.step, action)
            self.agent.update_from_env(next_obs, reward, done, info)

            if step >= self.max_steps:
                raise TerminationEvent(TerminationReason.MAX_TURNS_REACHED)
            if done:
                raise TerminationEvent(TerminationReason.ENV_DONE)

        raise TerminationEvent(TerminationReason.MAX_TURNS_REACHED)
