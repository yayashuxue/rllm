"""Strands workflow for mathematical problem solving with Python code interpreter."""

from rllm.agents.agent import Episode
from rllm.integrations.strands import RLLMModel, StrandsAgent
from rllm.rewards.reward_fn import math_reward_fn
from rllm.workflows.workflow import Workflow, handle_termination
from python_math_tool import python_tool


class StrandsMathWorkflow(Workflow):
    """A workflow for solving mathematical problems using Strands agent with Python code interpreter."""
    
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
        
        # Create the StrandsAgent with Python code interpreter tool
        self.strands_agent = StrandsAgent(
            model=self.rllm_model,
            system_prompt=(
                "You are a mathematical problem solver with access to a Python code interpreter. "
                "When solving problems:\n"
                "1. Read the problem carefully and understand what is being asked\n"
                "2. Break down complex problems into smaller steps\n"
                "3. Use Python code to perform calculations, solve equations, and verify your work\n"
                "4. For final answers, make sure to format them clearly\n"
                "5. If the problem asks for a specific format (like \\boxed{}), use that format\n"
                "6. Double-check your calculations using Python when possible\n"
                "7. Show your reasoning and calculations step by step"
            ),
            tools=[python_tool],
            **strands_agent_config
        )
        
        # Store sampling parameters
        self.sampling_params = sampling_params
        
        # Use the proper math reward function
        self.evaluator = self._evaluate_math_response

    def _evaluate_math_response(self, trajectory, task):
        """Evaluate the mathematical response using the math reward function."""
        if not trajectory.steps:
            return 0.0
            
        # Get the final response from the last step
        final_step = trajectory.steps[-1]
        if hasattr(final_step, 'action') and final_step.action:
            # Extract the response text
            if hasattr(final_step.action, 'message'):
                response = str(final_step.action.message)
            else:
                response = str(final_step.action)
        else:
            response = final_step.model_response or ""
        
        # Prepare task info for math reward function
        task_info = {
            "problem": task.get("question", task.get("problem", "")),
            "ground_truth": task.get("ground_truth", task.get("answer", "")),
            "data_source": task.get("data_source", "math"),
        }
        
        # Use the math reward function to evaluate
        reward_output = math_reward_fn(task_info, response)
        return reward_output.reward

    @handle_termination
    async def __call__(self, task: dict, uid: str, **kwargs) -> Episode:
        """Execute the mathematical problem-solving workflow."""
        
        # Reset the agent's trajectory
        self.strands_agent.reset_trajectory(task=task)
        
        # Extract the problem/question from the task
        problem = task.get("question") or task.get("problem") or task.get("task") or str(task)
        
        # Add some context to help the agent understand it's a math problem
        prompt = f"Please solve this mathematical problem step by step:\n\n{problem}"
        
        # Run the Strands agent with the problem
        try:
            await self.strands_agent.invoke_async(prompt, **self.sampling_params)
        except Exception as e:
            # If there's an error, still create a trajectory with the error info
            self.strands_agent._finish_current_step(
                model_response=f"Error: {str(e)}",
                action=None,
                done=True
            )
        
        # Get the agent's trajectory and evaluate it
        trajectory = self.strands_agent.trajectory
        reward = self.evaluator(trajectory, task)
        
        # Set trajectory properties
        trajectory.reward = reward
        trajectory.task = task
        
        # Create and return the episode
        episode = Episode()
        episode.id = uid
        episode.trajectories.append(("strands_math_agent", trajectory))
        episode.task = task
        
        return episode

    def assign_episode_correctness(self, episode: Episode) -> None:
        """Assign correctness to the episode based on trajectory reward."""
        for agent_name, trajectory in episode.trajectories:
            if agent_name == "strands_math_agent":
                episode.is_correct = trajectory.reward > 0
                break
        else:
            episode.is_correct = False 