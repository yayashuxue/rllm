from strands_tools import http_request

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

        # Researcher Agent with web capabilities
        self.researcher_agent = StrandsAgent(
            model=self.rllm_model,
            system_prompt=(
                "You are a Researcher Agent that gathers information from the web. "
                "1. Determine if the input is a research query or factual claim "
                "2. Use your research tools (http_request, retrieve) to find relevant information "
                "3. Include source URLs and keep findings under 500 words"
            ),
            callback_handler=None,
            tools=[http_request]
        )

        # Analyst Agent for verification and insight extraction
        self.analyst_agent = StrandsAgent(
            model=self.rllm_model,
            callback_handler=None,
            system_prompt=(
                "You are an Analyst Agent that verifies information. "
                "1. For factual claims: Rate accuracy from 1-5 and correct if needed "
                "2. For research queries: Identify 3-5 key insights "
                "3. Evaluate source reliability and keep analysis under 400 words"
            ),
        )

        # Writer Agent for final report creation
        self.writer_agent = StrandsAgent(
            model=self.rllm_model,
            system_prompt=(
                "You are a Writer Agent that creates clear reports. "
                "1. For fact-checks: State whether claims are true or false "
                "2. For research: Present key insights in a logical structure "
                "3. Keep reports under 500 words with brief source mentions"
            )
        )

        self.sampling_params = sampling_params


    def run_research_workflow(self, user_input: str):
        # Step 1: Researcher Agent gathers web information
        researcher_response = self.researcher_agent(
            f"Research: '{user_input}'. Use your available tools to gather information from reliable sources.",
        )
        research_findings = str(researcher_response)

        # Step 2: Analyst Agent verifies facts
        analyst_response = self.analyst_agent(
            f"Analyze these findings about '{user_input}':\n\n{research_findings}",
        )
        analysis = str(analyst_response)

        # Step 3: Writer Agent creates report
        final_report = self.writer_agent(
            f"Create a report on '{user_input}' based on this analysis:\n\n{analysis}"
        )

        return final_report
    
    def evaluate_report(self, report: str) -> float:
        """Evaluate the report and return a reward."""
        # For now, always return 0 as requested by the user
        # In the future, this could implement actual evaluation logic like:
        # - Checking if the report is accurate
        # - Measuring report quality
        return 0.0

    @handle_termination
    async def __call__(self, task: dict, uid: str, **kwargs) -> Episode:
        """Execute a multi-step workflow"""        
        # Extract the actual prompt/question from the task
        prompt = task.get("task") or task.get("question") or str(task)

        final_report = self.run_research_workflow(prompt)
        reward = self.evaluate_report(final_report)
        
        episode = Episode()
        episode.id = uid  # Set the episode ID to avoid KeyError in parallel execution
        episode.trajectories.append(("research_agent", self.researcher_agent.trajectory))
        episode.trajectories.append(("analyst_agent", self.analyst_agent.trajectory))
        episode.trajectories.append(("writer_agent", self.writer_agent.trajectory))
        episode.task = task
        episode.reward = reward

        return episode