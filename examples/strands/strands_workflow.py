from rllm.agents.agent import Episode
from rllm.integrations.strands import RLLMModel, StrandsAgent
from rllm.workflows.workflow import Workflow, handle_termination


class StrandsWorkflow(Workflow):
    def __init__(
        self,
        model_id: str = "gpt-4",
        model_config: dict = None,
        strands_agent_config: dict = None,
        sampling_params: dict = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Initialize mutable defaults
        model_config = dict(model_config) if model_config is not None else {}
        strands_agent_config = dict(strands_agent_config) if strands_agent_config is not None else {}
        sampling_params = dict(sampling_params) if sampling_params is not None else {}
        
        # Create the RLLMModel that wraps our rollout engine
        self.rllm_model = RLLMModel(
            rollout_engine=self.rollout_engine, 
            model_id=model_id,
            **model_config
        )
        
        # Create the StrandsAgent with the RLLMModel
        self.strands_agent = StrandsAgent(
            model=self.rllm_model,
            **strands_agent_config
        )
        
        # Register the agent (though StrandsAgent may not inherit from BaseAgent)
        # We'll handle trajectory collection manually
        self.sampling_params = sampling_params
        
        # Create the evaluator
        self.evaluator = lambda trajectory, task: 0.0

    @handle_termination
    async def __call__(self, task: dict, uid: str, **kwargs) -> Episode:
        """Execute a multi-step workflow"""
        
        self.strands_agent.reset_trajectory(task=task)
        
        # Extract the actual prompt/question from the task
        prompt = task.get("task") or task.get("question") or str(task)
        
        # Run the Strands agent with the prompt
        try:
            await self.strands_agent.invoke_async(prompt, **self.sampling_params)
        except Exception as e:
            # If there's an error, still create a trajectory with the error info
            self.strands_agent._finish_current_step(
                model_response=f"Error: {str(e)}",
                action=None,
                done=True
            )
        
        # Get the agent's trajectory
        trajectory = self.strands_agent.trajectory
        reward = self.evaluator(trajectory, task)

        trajectory.reward = reward
        trajectory.task = task
        
        episode = Episode()
        episode.id = uid  # Set the episode ID to avoid KeyError in parallel execution
        episode.trajectories.append(("strands_agent", trajectory))  # trajectories is a list of tuples
        episode.task = task

        return episode

    def assign_episode_correctness(self, episode: Episode) -> None:
        """Assign correctness to the episode based on trajectory reward."""
        # For now, episode is correct if reward > 0
        # Since we're always returning 0, this will always be False
        for agent_name, trajectory in episode.trajectories:
            if agent_name == "strands_agent":
                episode.is_correct = trajectory.reward > 0
                break