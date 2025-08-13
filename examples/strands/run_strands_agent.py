
import asyncio
import os
from dotenv import load_dotenv, find_dotenv

from rllm.engine.rollout_engine import RolloutEngine
from rllm.integrations.strands import RLLMModel, StrandsAgent


async def main():
    """Main example function."""
    print("=== Strands + rLLM Integration Example ===\n")
    load_dotenv(find_dotenv())

    from transformers import AutoTokenizer
    tokenizer_model = os.getenv("TOKENIZER_MODEL", "Qwen/Qwen3-4B")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    openai_kwargs = {}
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    if base_url:
        openai_kwargs["base_url"] = base_url
    if api_key:
        openai_kwargs["api_key"] = api_key

    rollout_engine = RolloutEngine(
        engine_name="openai",
        tokenizer=tokenizer,
        openai_kwargs=openai_kwargs,
    )
    
    # Manual creation (for more control)
    rollout_model = RLLMModel(
        rollout_engine=rollout_engine,
        model_id=os.getenv("MODEL_NAME", "gpt-4o"),
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
