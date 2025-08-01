import os

os.environ["OTEL_SDK_DISABLED"] = "true"

import asyncio

from rllm.engine.rollout_engine import RolloutEngine
from rllm.integrations.strands import RLLMModel, StrandsAgent


async def main():
    """Main example function."""
    print("=== Strands + rLLM Integration Example ===\n")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    rollout_engine = RolloutEngine(
        engine_name="openai",
        tokenizer=tokenizer,
        openai_kwargs={
            "base_url": "http://localhost:30000/v1",  # Local model server
            "api_key": "",
        },
    )
    
    
    # Method 2: Manual creation (for more control)
    rollout_model = RLLMModel(
        rollout_engine=rollout_engine,
        model_id="gpt-3.5-turbo",
        max_tokens=150,
        temperature=0.7
    )
    
    agent = StrandsAgent(model=rollout_model)
    print(f"Manually created agent with config: {agent.model.get_config()}")
    agent_response = agent("What is the capital of France?")
    print(agent_response)
    
    print("\n=== Example completed ===")


if __name__ == "__main__":    
    asyncio.run(main())
