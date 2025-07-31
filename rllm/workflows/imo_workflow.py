# Adapted from https://github.com/lyang36/IMO25/blob/main/code/agent.py

from copy import deepcopy

from rllm.agents.agent import Action, Episode
from rllm.agents.math_agent import MathAgent
from rllm.workflows.solve_verify_workflow import SolveVerifyWorkflow
from rllm.workflows.workflow import TerminationEvent, TerminationReason, handle_termination


class IMOWorkflow(SolveVerifyWorkflow):
    def __init__(
        self,
        solver_cls,
        verifier_cls,
        env_cls,
        solver_args=None,
        verifier_args=None,
        env_args=None,
        sampling_params=None,
        **kwargs,
    ):
        super(SolveVerifyWorkflow, self).__init__(**kwargs)

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

        # a stateless agent to determine if a solution claims it's complete
        self._checker = MathAgent(accumulate_thinking=False)
        self.register_agent(self._checker)

        self.env = env_cls(**env_args)
        self.sampling_params = sampling_params

        self.completed_trajectories = []

    @handle_termination
    async def __call__(self, task: dict, uid: str, **kwargs) -> Episode:
        """Execute a multi-step workflow"""

        observation, info = await self.run_in_executor(self.reset, task=task, uid=uid)  # returns observation and info from the environment
        question = observation["question"]

        solution = await self.initial_solver_turn(question=question, info=info)
        bug_report, good_verification = await self.verify_solution(question=question, solution=solution.action, info=info)

        correct_count = 1 if good_verification else 0
        error_count = 0

        for _ in range(30):
            if not good_verification:
                correct_count = 0
                error_count += 1

                solution = await self.solver_turn(question=question, solution=solution.action, bug_report=bug_report, info=info)

            bug_report, good_verification = await self.verify_solution(question=question, solution=solution.action, info=info)

            if good_verification:
                correct_count += 1
                error_count = 0

            if correct_count >= 5:
                raise TerminationEvent(TerminationReason.ENV_DONE)

            if error_count >= 10:
                raise TerminationEvent(TerminationReason.MAX_TURNS_EXCEEDED)

        raise TerminationEvent(TerminationReason.MAX_TURNS_EXCEEDED)

    async def stateless_model_call(self, prompt: str) -> bool:
        self._checker.reset()
        self._checker.update_from_env(prompt, 0, False, {})
        response = await self.get_model_response(self._checker, **self.sampling_params)
        return "yes" in response.lower()

    async def initial_solver_turn(self, question: str, info: dict) -> Action:
        self.solver.reset()
        self.solver.messages = [{"role": "system", "content": step1_prompt}]
        self.solver.update_from_env(question, 0, False, info)
        response = await self.get_model_response(self.solver, **self.sampling_params)
        solution = self.solver.update_from_model(response)
        _, reward, _, info = await self.run_in_executor(self.env.step_solver, solution)

        self.solver.update_from_env(self_improvement_prompt, reward, False, info)
        response = await self.get_model_response(self.solver, **self.sampling_params)
        solution = self.solver.update_from_model(response)
        _, reward, _, info = await self.run_in_executor(self.env.step_solver, solution)
        self.solver.update_from_env(None, reward, False, info)

        self.commit("solver", self.solver, reset=True)

        is_complete = await self.stateless_model_call(check_complete_prompt.format(solution=solution.action))
        if not is_complete:
            raise TerminationEvent(TerminationReason.MAX_TURNS_EXCEEDED)

        return solution

    async def solver_turn(self, question: str, solution: str, bug_report: str, info: dict) -> Action:
        self.solver.reset()
        self.solver.messages = [
            {"role": "system", "content": step1_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": solution},
        ]

        self.solver.update_from_env(correction_prompt.format(bug_report=bug_report), 0, False, info)
        response = await self.get_model_response(self.solver, **self.sampling_params)
        solution = self.solver.update_from_model(response)
        _, reward, _, info = await self.run_in_executor(self.env.step_solver, solution)
        self.solver.update_from_env(None, reward, False, info)
        self.commit("solver", self.solver, reset=True)

        is_complete = await self.stateless_model_call(check_complete_prompt.format(solution=solution.action))
        if not is_complete:
            raise TerminationEvent(TerminationReason.MAX_TURNS_EXCEEDED)

        return solution

    async def verify_solution(self, question: str, solution: str, info: dict) -> bool:
        self.verifier.reset()
        self.verifier.messages = [{"role": "system", "content": verification_system_prompt}]
        detailed_solution = extract_detailed_solution(solution)
        verification_prompt = verification_user_prompt.format(question=question, solution=detailed_solution, verification_remider=verification_remider)

        self.verifier.update_from_env(verification_prompt, 0, False, info)
        response = await self.get_model_response(self.verifier, **self.sampling_params)
        verification = self.verifier.update_from_model(response)
        _, reward, _, info = await self.run_in_executor(self.env.step_verifier, verification)
        self.verifier.update_from_env(None, reward, False, info)
        self.commit("verifier", self.verifier, reset=True)

        bug_report = ""
        good_verification = await self.stateless_model_call(check_correct_prompt.format(verification=verification.action))
        if not good_verification:
            bug_report = extract_detailed_solution(verification.action, "Detailed Verification", False)

        return bug_report, good_verification

    def collect_trajectories(self) -> Episode:
        episode = Episode()
        episode.trajectories = deepcopy(self.completed_trajectories)
        return episode

    def assign_episode_correctness(self, episode: Episode) -> None:
        solve_traj = [traj for agent_name, traj in episode.trajectories if agent_name == "solver"]
        last_solver_traj = solve_traj[-1]
        episode.is_correct = last_solver_traj.reward > 0


def extract_detailed_solution(solution, marker="Detailed Solution", after=True):
    """
    Extracts the text after '### Detailed Solution ###' from the solution string.
    Returns the substring after the marker, stripped of leading/trailing whitespace.
    If the marker is not found, returns an empty string.
    """
    idx = solution.find(marker)
    if idx == -1:
        return ""
    if after:
        return solution[idx + len(marker) :].strip()
    else:
        return solution[:idx].strip()


step1_prompt = """
### Core Instructions ###

*   **Rigor is Paramount:** Your primary goal is to produce a complete and rigorously justified solution. Every step in your solution must be logically sound and clearly explained. A correct final answer derived from flawed or incomplete reasoning is considered a failure.
*   **Honesty About Completeness:** If you cannot find a complete solution, you must **not** guess or create a solution that appears correct but contains hidden flaws or justification gaps. Instead, you should present only significant partial results that you can rigorously prove. A partial result is considered significant if it represents a substantial advancement toward a full solution. Examples include:
    *   Proving a key lemma.
    *   Fully resolving one or more cases within a logically sound case-based proof.
    *   Establishing a critical property of the mathematical objects in the problem.
    *   For an optimization problem, proving an upper or lower bound without proving that this bound is achievable.
*   **Use TeX for All Mathematics:** All mathematical variables, expressions, and relations must be enclosed in TeX delimiters (e.g., `Let $n$ be an integer.`).

### Output Format ###

Your response MUST be structured into the following sections, in this exact order.

**1. Summary**

Provide a concise overview of your findings. This section must contain two parts:

*   **a. Verdict:** State clearly whether you have found a complete solution or a partial solution.
    *   **For a complete solution:** State the final answer, e.g., "I have successfully solved the problem. The final answer is..."
    *   **For a partial solution:** State the main rigorous conclusion(s) you were able to prove, e.g., "I have not found a complete solution, but I have rigorously proven that..."
*   **b. Method Sketch:** Present a high-level, conceptual outline of your solution. This sketch should allow an expert to understand the logical flow of your argument without reading the full detail. It should include:
    *   A narrative of your overall strategy.
    *   The full and precise mathematical statements of any key lemmas or major intermediate results.
    *   If applicable, describe any key constructions or case splits that form the backbone of your argument.

**2. Detailed Solution**

Present the full, step-by-step mathematical proof. Each step must be logically justified and clearly explained. The level of detail should be sufficient for an expert to verify the correctness of your reasoning without needing to fill in any gaps. This section must contain ONLY the complete, rigorous proof, free of any internal commentary, alternative approaches, or failed attempts.

### Self-Correction Instruction ###

Before finalizing your output, carefully review your "Method Sketch" and "Detailed Solution" to ensure they are clean, rigorous, and strictly adhere to all instructions provided above. Verify that every statement contributes directly to the final, coherent mathematical argument.

"""

self_improvement_prompt = """
You have an opportunity to improve your solution. Please review your solution carefully. Correct errors and fill justification gaps if any. Your second round of output should strictly follow the instructions in the system prompt.
"""

correction_prompt = """
Below is the bug report. If you agree with certain item in it, can you improve your solution so that it is complete and rigorous? Note that the evaluator who generates the bug report can misunderstand your solution and thus make mistakes. If you do not agree with certain item in the bug report, please add some detailed explanations to avoid such misunderstanding. Your new solution should strictly follow the instructions in the system prompt.

{bug_report}
"""

verification_system_prompt = """
You are an expert mathematician and a meticulous grader for an International Mathematical Olympiad (IMO) level exam. Your primary task is to rigorously verify the provided mathematical solution. A solution is to be judged correct **only if every step is rigorously justified.** A solution that arrives at a correct final answer through flawed reasoning, educated guesses, or with gaps in its arguments must be flagged as incorrect or incomplete.

### Instructions ###

**1. Core Instructions**
*   Your sole task is to find and report all issues in the provided solution. You must act as a **verifier**, NOT a solver. **Do NOT attempt to correct the errors or fill the gaps you find.**
*   You must perform a **step-by-step** check of the entire solution. This analysis will be presented in a **Detailed Verification Log**, where you justify your assessment of each step: for correct steps, a brief justification suffices; for steps with errors or gaps, you must provide a detailed explanation.

**2. How to Handle Issues in the Solution**
When you identify an issue in a step, you MUST first classify it into one of the following two categories and then follow the specified procedure.

*   **a. Critical Error:**
    This is any error that breaks the logical chain of the proof. This includes both **logical fallacies** (e.g., claiming that `A>B, C>D` implies `A-C>B-D`) and **factual errors** (e.g., a calculation error like `2+3=6`).
    *   **Procedure:**
        *   Explain the specific error and state that it **invalidates the current line of reasoning**.
        *   Do NOT check any further steps that rely on this error.
        *   You MUST, however, scan the rest of the solution to identify and verify any fully independent parts. For example, if a proof is split into multiple cases, an error in one case does not prevent you from checking the other cases.

*   **b. Justification Gap:**
    This is for steps where the conclusion may be correct, but the provided argument is incomplete, hand-wavy, or lacks sufficient rigor.
    *   **Procedure:**
        *   Explain the gap in the justification.
        *   State that you will **assume the step's conclusion is true** for the sake of argument.
        *   Then, proceed to verify all subsequent steps to check if the remainder of the argument is sound.

**3. Output Format**
Your response MUST be structured into two main sections: a **Summary** followed by the **Detailed Verification Log**.

*   **a. Summary**
    This section MUST be at the very beginning of your response. It must contain two components:
    *   **Final Verdict**: A single, clear sentence declaring the overall validity of the solution. For example: "The solution is correct," "The solution contains a Critical Error and is therefore invalid," or "The solution's approach is viable but contains several Justification Gaps."
    *   **List of Findings**: A bulleted list that summarizes **every** issue you discovered. For each finding, you must provide:
        *   **Location:** A direct quote of the key phrase or equation where the issue occurs.
        *   **Issue:** A brief description of the problem and its classification (**Critical Error** or **Justification Gap**).

*   **b. Detailed Verification Log**
    Following the summary, provide the full, step-by-step verification log as defined in the Core Instructions. When you refer to a specific part of the solution, **quote the relevant text** to make your reference clear before providing your detailed analysis of that part.

**Example of the Required Summary Format**
*This is a generic example to illustrate the required format. Your findings must be based on the actual solution provided below.*

**Final Verdict:** The solution is **invalid** because it contains a Critical Error.

**List of Findings:**
*   **Location:** "By interchanging the limit and the integral, we get..."
    *   **Issue:** Justification Gap - The solution interchanges a limit and an integral without providing justification, such as proving uniform convergence.
*   **Location:** "From $A > B$ and $C > D$, it follows that $A-C > B-D$"
    *   **Issue:** Critical Error - This step is a logical fallacy. Subtracting inequalities in this manner is not a valid mathematical operation.

"""


verification_remider = """
### Verification Task Reminder ###

Your task is to act as an IMO grader. Now, generate the **summary** and the **step-by-step verification log** for the solution above. In your log, justify each correct step and explain in detail any errors or justification gaps you find, as specified in the instructions above.
"""

check_complete_prompt = """
Is the following text claiming that the solution is complete?
==========================================================

{solution}

==========================================================

Response in exactly "yes" or "no". No other words.
"""

check_correct_prompt = """
Response in "yes" or "no". Is the following statement saying the solution is correct, or does not contain critical error or a major justification gap?

{verification}
"""

verification_user_prompt = """
======================================================================
### Problem ###

{question}

======================================================================
### Solution ###

{solution}

{verification_remider}
"""
