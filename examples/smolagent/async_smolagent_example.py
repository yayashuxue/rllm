#!/usr/bin/env python3
"""
Example script demonstrating async run functionality for SmolAgent with real RolloutEngine.

This script shows how to use AsyncCodeAgent and AsyncToolCallingAgent with the arun method
for full async agent execution with a real model API endpoint.

Key Features:
- Uses RLLMOpenAIModel which properly converts SmolAgent messages to OpenAI format
- Tool responses (MessageRole.TOOL_RESPONSE) are correctly converted to "role": "tool"
- Full async execution with arun() method for complete task resolution
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer

from rllm.engine.rollout_engine import RolloutEngine
from rllm.integrations.smolagents import AsyncCodeAgent, AsyncToolCallingAgent, RLLMOpenAIModel


async def test_async_code_agent(rollout_engine):
    """Test AsyncCodeAgent with a real RolloutEngine."""
    print("\n" + "="*60)
    print("🤖 Testing AsyncCodeAgent")
    print("="*60)
    
    # Create model with real rollout engine
    model = RLLMOpenAIModel(rollout_engine=rollout_engine, sampling_params={"max_tokens": 1000})
    
    # Create async code agent
    async_code_agent = AsyncCodeAgent(
        tools=[],
        model=model,
        max_steps=5,
        additional_authorized_imports=["math", "random", "json"],  # Only include commonly available modules
        add_base_tools=False  # Disable base tools to avoid dependency issues
    )
    
    # Test with a mathematical calculation task
    print("📝 Task: Calculate the area of a circle with radius 5 using math module")
    
    try:
        # Execute async run (full agent execution)
        result = await async_code_agent.arun("Calculate the area of a circle with radius 5 using the math module and return the result")
        
        print("✅ AsyncCodeAgent arun completed successfully!")
        print(f"📊 Result type: {type(result)}")
        print(f"🎯 Final Answer: {result}")
            
        return True
        
    except Exception as e:
        print(f"❌ AsyncCodeAgent test failed: {e}")
        return False


async def test_async_tool_calling_agent(rollout_engine):
    """Test AsyncToolCallingAgent with a real RolloutEngine."""
    print("\n" + "="*60)
    print("🛠️  Testing AsyncToolCallingAgent")
    print("="*60)
    
    # Create a simple calculator tool
    from smolagents import Tool
    
    class CalculatorTool(Tool):
        name = "calculator"
        description = "Performs basic mathematical calculations"
        inputs = {
            "expression": {
                "type": "string", 
                "description": "Mathematical expression to evaluate (e.g., '2+2', '3*4')"
            }
        }
        output_type = "number"
        
        def forward(self, expression: str) -> float:
            """Safely evaluate a mathematical expression."""
            try:
                # Simple safety check - only allow basic math operations
                allowed_chars = set("0123456789+-*/.() ")
                if not all(c in allowed_chars for c in expression):
                    return "Error: Only basic math operations are allowed"
                
                result = eval(expression)
                return float(result)
            except Exception as e:
                return f"Error: {str(e)}"
    
    calculator = CalculatorTool()
    
    # Create model with real rollout engine
    model = RLLMOpenAIModel(rollout_engine=rollout_engine)
    
    # Create async tool calling agent
    async_tool_agent = AsyncToolCallingAgent(
        tools=[calculator],
        model=model,
        max_steps=5,
        add_base_tools=False  # Disable base tools to avoid dependency issues
    )
    
    print("📝 Task: Use the calculator tool to compute 15 * 8 + 20")
    
    try:
        # Execute async run (full agent execution)
        result = await async_tool_agent.arun(
            "Use the calculator tool to compute 15 * 8 + 20 and provide the final answer"
        )
        
        print("✅ AsyncToolCallingAgent arun completed successfully!")
        print(f"📊 Result type: {type(result)}")
        print(f"🎯 Final Answer: {result}")
            
        return True
        
    except Exception as e:
        print(f"❌ AsyncToolCallingAgent test failed: {e}")
        return False


async def main():
    """Main function to run the async SmolAgent examples."""
    print("🚀 Async SmolAgent Example with Real RolloutEngine")
    print("=" * 60)
    
    # Initialize tokenizer and rollout engine (similar to the attached example)
    print("🔧 Initializing RolloutEngine...")
    
    model_name = "Qwen/Qwen3-4B"  # Using a smaller model for faster testing
    print(f"📥 Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    rollout_engine = RolloutEngine(
        engine_name="openai",
        tokenizer=tokenizer,
        openai_kwargs={
            "base_url": "http://localhost:30000/v1",  # Local model server
            "api_key": "None",
        },
        disable_thinking=False,
    )
    
    print("✅ RolloutEngine initialized successfully!")
        
    # Test both async agents
    success_count = 0
    total_tests = 2
    
    # Test AsyncCodeAgent
    if await test_async_code_agent(rollout_engine):
        success_count += 1
    
    # Test AsyncToolCallingAgent
    if await test_async_tool_calling_agent(rollout_engine):
        success_count += 1
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    print(f"✅ Tests passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 All async agent runs working correctly!")
    else:
        print("⚠️  Some tests failed - check the error messages above")
    
    print("\n🔍 Key Features Demonstrated:")
    print("  • AsyncCodeAgent.arun(): Full async code generation and execution")
    print("  • AsyncToolCallingAgent.arun(): Full async tool calling and execution")
    print("  • RLLMOpenAIModel: Integration with real model APIs")
    print("  • Real RolloutEngine: Production-ready model interface")
    print("  • Multi-step async execution: Complete task resolution")
    print("  • Error handling: Robust error management")


if __name__ == "__main__":
    print("🐍 Python Async SmolAgent Example - arun() Method")
    print("Make sure you have a local model server running!")
    print("Example: python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-3B --port 30000")
    print()
    
    asyncio.run(main())