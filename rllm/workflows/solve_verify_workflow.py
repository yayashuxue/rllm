from copy import deepcopy

from rllm.agents.agent import Action, BaseAgent, Episode
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow, handle_termination


class SolveVerifyWorkflow(Workflow):
    def __init__(
        self,
        solver_cls,
        verifier_cls,
        env_cls,
        solver_args=None,
        verifier_args=None,
        env_args=None,
        sampling_params=None,
        num_initial_solver_iters=2,
        max_verify_solver_iters=10,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Initialize mutable defaults
        solver_args = dict(solver_args) if solver_args is not None else {}
        verifier_args = dict(verifier_args) if verifier_args is not None else {}
        env_args = dict(env_args) if env_args is not None else {}
        sampling_params = dict(sampling_params) if sampling_params is not None else {}

        self.solver = solver_cls(**solver_args)
        self.verifier = verifier_cls(**verifier_args)

        # both use the default rollout engine (from the workflow engine)
        self.register_agent(self.solver)
        self.register_agent(self.verifier)

        self.env = env_cls(**env_args)
        self.sampling_params = sampling_params

        self.num_initial_solver_iters = num_initial_solver_iters
        self.max_verify_solver_iters = max_verify_solver_iters

        self.completed_trajectories = []

    @handle_termination
    async def __call__(self, task: dict, uid: str, **kwargs) -> Episode:
        """Execute a multi-step workflow"""

        observation, info = await self.run_in_executor(self.reset, task=task, uid=uid)  # returns observation and info from the environment

        action = None
        for i in range(self.num_initial_solver_iters):
            # solve step (inspired by https://github.com/lyang36/IMO25/blob/main/IMO25.pdf and https://matharena.ai/imo/)
            solver_prompt = self._build_solver_prompt(observation=observation, action=action)
            self.solver.update_from_env(solver_prompt, 0, False, info)
            response = await self.get_model_response(self.solver, **self.sampling_params)
            action = self.solver.update_from_model(response)
            _, reward, _, info = await self.run_in_executor(self.env.step_solver, action)
            self.solver.update_from_env(None, reward, False, info)

        self.prune_history(self.solver)  # only keep the solver's latest solution

        for i in range(self.max_verify_solver_iters):
            # verification step (inspired by https://arxiv.org/abs/2504.10337 and inspired by https://github.com/lyang36/IMO25/blob/main/IMO25.pdf)
            verifier_prompt = self._build_verifier_prompt(task=task, action=action)
            self.verifier.update_from_env(verifier_prompt, 0, False, info)
            response = await self.get_model_response(self.verifier, **self.sampling_params)
            verification = self.verifier.update_from_model(response)
            # verifier is rewarded if the verification is correct wrt to the action
            # env throws done after self.env_args.max_consecutive_correct_verifications consecutive "correct" verifications
            _, reward, done, info = await self.run_in_executor(self.env.step_verifier, verification)
            self.verifier.update_from_env(None, reward, done, info)
            self.commit("verifier", self.verifier, reset=True)

            # the env stopping condition is based on the number of recent positive verifications
            if done:
                raise TerminationEvent(TerminationReason.ENV_DONE)

            # solver step (inspired by https://github.com/lyang36/IMO25/blob/main/IMO25.pdf and https://matharena.ai/imo/)
            solver_prompt = self._build_solver_prompt(verification=verification)
            self.solver.update_from_env(solver_prompt, 0, False, info)
            response = await self.get_model_response(self.solver, **self.sampling_params)
            action = self.solver.update_from_model(response)
            _, reward, _, info = await self.run_in_executor(self.env.step_solver, action)
            self.solver.update_from_env(None, reward, False, info)

        raise TerminationEvent(TerminationReason.MAX_TURNS_EXCEEDED)

    def commit(self, name: str, agent: BaseAgent, reset: bool = False) -> None:
        traj = agent.trajectory
        if traj.steps:
            self.completed_trajectories.append((name, deepcopy(traj)))
        if reset:
            agent.reset()

    def collect_trajectories(self) -> Episode:
        episode = Episode()
        episode.trajectories = deepcopy(self.completed_trajectories)
        episode.trajectories.append(("solver", deepcopy(self.solver.trajectory)))
        return episode

    def assign_episode_correctness(self, episode: Episode) -> None:
        total_reward = 0
        for agent_name, trajectory in episode.trajectories:
            if agent_name == "verifier":
                continue
            total_reward += trajectory.reward
        episode.is_correct = total_reward > 0

    def prune_history(self, agent: BaseAgent) -> None:
        """Updates agent.messages to include only the first and last message."""
        if len(agent.messages) > 2:
            initial_msg = agent.messages[0]
            final_msg = agent.messages[-1]
            agent.messages = [initial_msg, final_msg]

    def _build_solver_prompt(self, observation: dict | None = None, action: Action | None = None, verification: Action | None = None) -> str:
        SOLVER_PROMPT = """
You are an expert mathematician. Your job is to solve a challenging competition-level math problem by producing one complete, rigorous, self-contained solution.

## Problem
{problem}

## Instructions
1. Think through the problem step by step. If you get stuck, try to backtrack or consider an alternative approach.
2. Present the solution:
    - Write a cohesive proof or derivation. Every step must be logically sound, clearly explained, and necessary.
    - In each step, state what is done, why it is done, and why it is allowed.
    - Number each major logical step.
    - Do not skip steps or handwave. 
    - A correct solution derived from flawed or incomplete reasoning will be considered incorrect.
3. You can use general theorems and lemmas, but only if they are well-known. As a rule of thumb: if the result has a name and is famous enough to have a Wikipedia page or something similar to describe it, it is allowed. Any result from papers that would not be taught in high school or low-level bachelor courses in mathematics should not be used.
4. Use correct LaTeX notation to write equations and mathematical symbols. You should encompass these equations in appropriate symbols ("\\(" and "\\)" for inline math, "\\[" and "\\]" for block math). Do not use any Unicode characters. 
5. Present your final solution as a self-contained derivation or proof. If the problem requires a specific final answer, enclose it in \\boxed{{}}. If the task is to prove a statement, end your solution with an empty \\boxed{{}}.
""".strip()

        SOLVER_RETRY_PROMPT = """
You have been given another attempt to solve the problem. Please review and revise your previous solution accordingly. If your previous solution was unfinished, you should finish it. Follow the same instructions as before.
""".strip()

        SOLVER_REFINE_PROMPT = """
You are an expert mathematician. A verification report for your current solution is provided below. Revise and perfect your existing solution accordingly.

## Verification
{verification}

## Instructions
1. Analyze the verification report:
    - Summarize the verifier's key findings: which steps were correct, which had flaws (and how severe).
    - List any missing or invalid assumptions the verifier identified.
    - Outline a plan to fix each flaw or fill gaps from first principles.
2. Rewrite the solution:
    - Write a brand-new, coherent solution from scratch. Every step must be logically sound, clearly explained, and necessary.
    - In each step, state what is done, why it is done, and why it is allowed.
    - Number each major logical step.
    - Do not skip steps or handwave.
    - A correct solution derived from flawed or incomplete reasoning will be considered incorrect.
3. You can use general theorems and lemmas, but only if they are well-known. As a rule of thumb: if the result has a name and is famous enough to have a Wikipedia page or something similar to describe it, it is allowed. Any result from papers that would not be taught in high school or low-level bachelor courses in mathematics should not be used.
4. Use correct LaTeX notation to write equations and mathematical symbols. You should encompass these equations in appropriate symbols ("\\(" and "\\)" for inline math, "\\[" and "\\]" for block math). Do not use any Unicode characters. 
5. Present your final solution as a self-contained derivation or proof. If the problem requires a specific final answer, enclose it in \\boxed{{}}. If the task is to prove a statement, end your solution with an empty \\boxed{{}}.
""".strip()

        if observation and not action and not verification:
            return SOLVER_PROMPT.format(problem=observation.get("question", ""))
        elif observation and action and not verification:
            return SOLVER_RETRY_PROMPT
        elif verification:
            verification = self._strip_grading_from_verification(verification.action)
            return SOLVER_REFINE_PROMPT.format(verification=verification)
        else:
            raise ValueError("Invalid input to _build_solver_prompt")

    def _build_verifier_prompt(self, task: dict, action: Action) -> str:
        VERIFIER_PROMPT = """
You are an expert mathematician. Your job is to audit a given solution to a challenging competition-level math problem.

## Problem
{problem}

## Solution
{solution}

## Instructions
1. Read the solution and number its individual logical steps (e.g., "Step 1: ...", "Step 2: ...").
2. Step-by-step audit: for each numbered step, do the following:
    - Forward check: Verify that the conclusion of this step follows correctly from the previous step(s).  
    - Justification: Assess whether the reasoning or citation (theorem, definition, etc.) is adequate. Ensure any theorems and lemmas used are well-known. As a rule of thumb: if the result has a name and is famous enough to have a Wikipedia page or something similar to describe it, it is allowed. Any result from papers that would not be taught in high school or low-level bachelor courses in mathematics should not be used.  
    - Assumptions: Confirm all hidden or stated assumptions (domains, nonzero denominators, continuity, etc.) are valid.
    - Issues: If you identify any error or gap, describe precisely where it occurs, why it's wrong or incomplete, and how severe it is.
3. Backward check: After the final step, verify that the claimed result indeed satisfies the original problem statement under all required conditions.
4. Do NOT repair or extend the solution. Your job is to audit the solution, not to fix it.
5. Use correct LaTeX notation to write equations and mathematical symbols. You should encompass these equations in appropriate symbols ("\\(" and "\\)" for inline math, "\\[" and "\\]" for block math). Do not use any Unicode characters.
6. Present your audit as a self-contained, numbered walkthrough of the solution's steps. If you believe the solution's final answer is correct, even if the solution has minor flaws, end your verification with \\boxed{{1}}. Otherwise, end your verification with \\boxed{{0}}.
""".strip()

        return VERIFIER_PROMPT.format(problem=task.get("question", ""), solution=action.action)

    def _strip_grading_from_verification(self, verification_text: str) -> str:
        """Strip out the grading part (\boxed{0} or \boxed{1}) from verification output."""
        from rllm.rewards.math_utils.utils import last_boxed_only_string

        # Find the last occurrence of \boxed{0} or \boxed{1}
        last_boxed = last_boxed_only_string(verification_text)
        if last_boxed and last_boxed in ["\\boxed{0}", "\\boxed{1}"]:
            # Remove the last occurrence from the text
            return verification_text[: verification_text.rfind(last_boxed)].strip()
        return verification_text.strip()
