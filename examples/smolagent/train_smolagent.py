import hydra

from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer
from rllm.workflows.smolagent_workflow import SmolAgentWorkflow
from rllm.integrations.smolagents import AsyncCodeAgent
from smolagents import WebSearchTool

@hydra.main(config_path="pkg://rllm.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("deepscaler_math", "train")
    test_dataset = DatasetRegistry.load_dataset("aime2024", "test")
    trainer = AgentTrainer(
        workflow_class=SmolAgentWorkflow,
        workflow_args={
            "agent_cls": AsyncCodeAgent,
            "agent_args": {"tools": [WebSearchTool()]},
            "max_prompt_length": config.data.max_prompt_length,
            "max_response_length": config.data.max_response_length,
            "sampling_params": None,
            "max_steps": 2,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
