import hydra

from rllm.agents.math_agent import MathAgent
from rllm.data.dataset import DatasetRegistry
from rllm.environments.verify.solve_verify_env import SolveVerifyEnvironment
from rllm.rewards.reward_fn import math_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer
from rllm.workflows.solve_verify_workflow import SolveVerifyWorkflow


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("deepscaler_math", "train")
    test_dataset = DatasetRegistry.load_dataset("aime2024", "test")

    trainer = AgentTrainer(
        workflow_class=SolveVerifyWorkflow,
        workflow_args={
            "solver_cls": MathAgent,
            "solver_args": {"accumulate_thinking": False},
            "verifier_cls": MathAgent,
            "verifier_args": {"accumulate_thinking": False},
            "env_cls": SolveVerifyEnvironment,
            "env_args": {"reward_fn": math_reward_fn, "max_consecutive_correct_verifications": 2},
            "max_prompt_length": config.data.max_prompt_length,
            "max_response_length": config.data.max_response_length,
            "sampling_params": None,
            "num_initial_solver_iters": 1,
            "max_verify_solver_iters": 1,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
