import hydra

from examples.countdown.countdown_agent import CountDownAgent
from examples.countdown.countdown_reward import countdown_reward_fn
from rllm.data.dataset import DatasetRegistry
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    # Get training dataset split from config, default to "train"
    train_split = getattr(config.data, 'train_split', 'train')
    
    train_dataset = DatasetRegistry.load_dataset("countdown", train_split)
    test_dataset = DatasetRegistry.load_dataset("countdown", "test")

    print(f"Using training dataset split: {train_split}")

    env_args = {"reward_fn": countdown_reward_fn}

    trainer = AgentTrainer(
        agent_class=CountDownAgent,
        agent_args={},
        env_args=env_args,
        env_class=SingleTurnEnvironment,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main() 