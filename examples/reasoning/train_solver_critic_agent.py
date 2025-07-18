import hydra

from rllm.agents.critique_agent import CritiqueAgent
from rllm.agents.math_agent import MathAgent
from rllm.data.dataset import DatasetRegistry
from rllm.environments.base.critique_env import CritiqueEnvironment
from rllm.rewards.reward_fn import math_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer
from rllm.workflows.critique_workflow import CritiqueWorkflow


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("deepscaler_math", "train")
    test_dataset = DatasetRegistry.load_dataset("aime2024", "test")

    trainer = AgentTrainer(
        workflow_class=CritiqueWorkflow,
        workflow_args={
            "solver_cls": MathAgent,
            "solver_args": {"accumulate_thinking": False},
            "critic_cls": CritiqueAgent,
            "critic_args": {"accumulate_thinking": False},
            "env_cls": CritiqueEnvironment,
            "env_args": {"reward_fn": math_reward_fn},
            "max_prompt_length": config.data.max_prompt_length,
            "max_response_length": config.data.max_response_length,
            "sampling_params": None,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
