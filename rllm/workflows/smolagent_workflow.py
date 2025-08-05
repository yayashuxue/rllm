from rllm.agents.agent import Episode
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow, handle_termination
import inspect
from rllm.integrations.smolagents import RLLMOpenAIModel
import random

class SmolAgentWorkflow(Workflow):
    def __init__(
        self,
        agent_cls,
        agent_args=None,
        sampling_params=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Initialize mutable defaults
        agent_args = dict(agent_args) if agent_args is not None else {}
        sampling_params = dict(sampling_params) if sampling_params is not None else {}
        if agent_cls.__name__ == "AsyncCodeAgent":
            model = RLLMOpenAIModel(rollout_engine=self.rollout_engine, sampling_params={"max_tokens": sampling_params.get("max_tokens", 1000)})
            agent_args["model"] = model
        
        self.agent = agent_cls(**agent_args)
        self.register_agent(self.agent)
        self.sampling_params = sampling_params
        self.max_steps = kwargs.get("max_steps", 5)

    @handle_termination
    async def __call__(self, task: dict, uid: str, **kwargs) -> Episode:
        self.task = task
        self.uid = uid

        problem = task["problem"]
        await self.agent.arun(problem, max_steps=self.max_steps)
        raise TerminationEvent(TerminationReason.ENV_DONE)

    def collect_trajectories(self) -> Episode:
        """Collect the trajectories from the workflow"""

        episode = Episode()
        # import pdb; pdb.set_trace()

        for attr_name in dir(self):
            # Skip private attributes and methods
            if attr_name.startswith("_"):
                continue

            attr_value = getattr(self, attr_name)

            # Check if attribute is a BaseAgent instance
            if hasattr(attr_value, "trajectory"):
                episode.trajectories.append((attr_name, attr_value.trajectory))

        assert len(episode.trajectories) > 0, "No trajectories found in the workflow"

        return episode

    def postprocess_episode(self, episode: Episode, termination_reason: TerminationReason = None) -> Episode:
        """Collect and process the trajectories"""
        assert episode is not None, "Remember to call collect_trajectories() before postprocessing the episode"

        # 1. assign a task id and task
        episode.id = self.uid
        episode.task = self.task


        for agent_name, trajectory in episode.trajectories:
            # TODO: assign based on reward function, now it's fake
            trajectory.steps[-1].reward = 1.0 * random.randint(0, 1)
            self.compute_trajectory_reward(agent_name, trajectory)

        self.assign_episode_correctness(episode)

        # 5. assign a termination reason
        episode.termination_reason = termination_reason

        return episode
