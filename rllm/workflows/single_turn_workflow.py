from rllm.agents.agent import Episode
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow, handle_termination


class SingleTurnWorkflow(Workflow):
    def __init__(
        self,
        agent_cls,
        env_cls,
        agent_args=None,
        env_args=None,
        sampling_params=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Initialize mutable defaults
        agent_args = dict(agent_args) if agent_args is not None else {}
        env_args = dict(env_args) if env_args is not None else {}
        sampling_params = dict(sampling_params) if sampling_params is not None else {}

        self.agent = agent_cls(**agent_args)
        self.register_agent(self.agent)
        self.env = env_cls(**env_args)
        self.sampling_params = sampling_params

    @handle_termination
    async def __call__(self, task: dict, uid: str, **kwargs) -> Episode:
        observation, info = await self.run_in_executor(self.reset, task=task, uid=uid)  # returns observation and info from the environment
        self.agent.update_from_env(observation, 0, False, info)

        response = await self.get_model_response(self.agent, **self.sampling_params)
        action = self.agent.update_from_model(response)

        next_obs, reward, done, info = await self.run_in_executor(self.env.step, action)
        self.agent.update_from_env(next_obs, reward, done, info)

        raise TerminationEvent(TerminationReason.ENV_DONE)
