# rLLM + Strands Integration Guide

This guide explains how rLLM integrates with the Strands agent framework to enable reinforcement learning training of web-capable agents.

## Overview

The rLLM + Strands integration bridges two powerful frameworks:

- **rLLM**: Provides RL training infrastructure (PPO, trajectory collection, reward assignment)
- **Strands**: Provides agent framework and tool ecosystem (browser automation, API calls, memory)

The key innovation is **StrandsAgent**, which inherits all Strands capabilities while adding rLLM's trajectory tracking for RL training.

## Architecture

### Component Responsibilities

| Component | Responsibility | Key Features |
|-----------|----------------|--------------|
| **rLLM** | RL Training Infrastructure | `Step`, `Trajectory` data structures<br/>PPO training pipeline<br/>Reward assignment system |
| **Strands** | Agent Framework | Tool execution (browser, APIs)<br/>Message management<br/>Agent decision logic |
| **StrandsAgent** | Bridge Layer | Inherits Strands functionality<br/>Adds RL trajectory tracking<br/>Routes through `invoke_async()` |

### Data Flow

```
User Input → StrandsAgent.invoke_async() → Strands Core Logic → Tool Execution → RL Step Recording
```

Each interaction creates a complete RL training sample:

```python
Step(
    observation="User input or environment state",
    model_response="LLM output text", 
    action="Tool call or decision made",
    reward=0.0,  # Assigned later by reward function
    done=False,  # Whether episode completed
    chat_completions=[...]  # Full conversation context
)
```

## Core Data Structures

### Step (rLLM)

```python
@dataclass
class Step:
    observation: Any = None        # Environment state (user input, browser state)
    model_response: str = ""       # Raw LLM output
    action: Any = None            # Action taken (tool call, response)
    reward: float = 0.0           # RL reward (assigned by reward function)
    done: bool = False            # Episode termination flag
    chat_completions: list = []    # Full conversation history
    info: dict = {}               # Additional metadata
```

### Trajectory (rLLM)

```python
@dataclass  
class Trajectory:
    task: Any = None              # Original task/query
    steps: list[Step] = []        # Sequence of interaction steps
    reward: float = 0.0          # Cumulative episode reward
```

## Implementation Details

### StrandsAgent Bridge

The `StrandsAgent` class extends Strands' base `Agent` with RL tracking:

```python
class StrandsAgent(Agent):  # Inherits all Strands functionality
    def __init__(self, model, **kwargs):
        super().__init__(model=model, **kwargs)
        self._trajectory = Trajectory()    # rLLM data structure
        self._current_step = None
    
    async def invoke_async(self, prompt):
        # 1. Start RL step tracking
        self._start_new_step(observation=prompt)
        
        # 2. Execute Strands agent logic (unchanged)
        result = await super().invoke_async(prompt)
        
        # 3. Complete RL step recording
        self._finish_current_step(
            model_response=extracted_response,
            action=executed_action,
            done=is_episode_complete
        )
        
        return result  # Return normal Strands result
```

### Mixed Dispatch System

The integration supports both modern tool_calls and text parsing fallback:

```python
# Try tool_calls first (structured)
if supports_tool_calls:
    response = await model.get_response(
        messages, tools=tool_specs, tool_choice="auto"
    )
    if response.tool_calls:
        # Execute through StrandsAgent for RL tracking
        return await agent.invoke_async(tool_call_prompt)

# Fallback to text parsing (always works)
response = await agent.invoke_async(user_prompt)
action = parse_action_from_text(response)
```

## Practical Example

### Browser Research Agent

Here's how a web research task creates RL training data:

```python
# Step 1: User asks question
observation = "Find information about climate change impacts"
agent._start_new_step(observation=observation)

# Step 2: Agent decides to search
model_response = "I need to search for climate change information"
action = {
    "tool_call": "browser", 
    "arguments": {
        "action": {"type": "navigate", "url": "https://search.com?q=climate+change"}
    }
}

# Step 3: Tool execution happens (through Strands)
browser_result = {"status": "success", "content": "Search results loaded"}

# Step 4: RL step completed
agent._finish_current_step(
    model_response=model_response,
    action=action,
    reward=0.0,  # Will be assigned later
    done=False
)

# Result: Complete RL training sample ready for PPO
```

### Training Integration

The collected trajectories integrate seamlessly with rLLM training:

```python
# After episode completion
trajectory = agent.trajectory

# Assign rewards (domain-specific logic)
for step in trajectory.steps:
    step.reward = reward_function(step.observation, step.action, step.info)

# Train with PPO
ppo_trainer.train([trajectory])
```

## Tool Integration Features

### Enhanced Browser Executor

The integration includes robust tool validation and error handling:

```python
class BrowserExecutor:
    def __init__(self, tool):
        self._tool = tool
        # Extract allowed actions from BrowserInput schema
        self._allowed = get_browser_actions_from_models()
    
    async def execute(self, args):
        # 1. Validate action type
        if action_type not in self._allowed:
            return {"error": f"Invalid action. Use: {self.allowed}"}
        
        # 2. Normalize missing fields
        normalized_args = self._normalize(args)
        
        # 3. Intelligent rewriting (search → navigate)
        if action_type == "search":
            return self._rewrite_to_navigate(args)
        
        # 4. Execute with error handling
        try:
            return await self._tool(normalized_args)
        except ValidationError as e:
            return {"error": "validation_error", "details": e.errors()}
```

### Tool Spec Conversion

Automatic conversion between Strands ToolSpec and OpenAI tool format:

```python
def _to_openai_tools(self, tool_specs):
    """Convert Strands ToolSpec to OpenAI chat tools format"""
    tools = []
    for spec in tool_specs:
        tools.append({
            "type": "function",
            "function": {
                "name": extract_name(spec),
                "description": extract_description(spec), 
                "parameters": extract_schema(spec)
            }
        })
    return tools
```

## Running the Example

### Setup

```bash
# Install dependencies
pip install -e .
pip install strands-tools

# Set environment variables
export OPENAI_API_KEY="your-key"
export BROWSER_HEADLESS="true"
export MAX_STEPS="5"
```

### Execute

```bash
cd examples/strands
python strands_rllm_browser_agent_v2.py
```

### Expected Output

```
=== Strands Browser Research (minimal) ===
Model: gpt-4o
Tools: ['browser']
Question: [Your research question]

[step 1] Querying model gpt-4o (prefer tool_calls)...
[step 1] Querying model gpt-4o (text fallback)...
[step 1] LLM output: I need to search for information...
[step 1] Dispatching action: browser args={"action": {"type": "navigate"}}
[step 1] Observation: {"status": "success", "content": "..."}

🧪 RL INTEGRATION VERIFICATION
✅ Trajectory exists: True
✅ Steps recorded: 3
✅ Total reward: 0.0
🎯 INTEGRATION SUCCESS: 100% (4/4 checks passed)
```

## Key Benefits

### For RL Training
- **Complete trajectory data**: Every interaction recorded with full context
- **Reward assignment ready**: Infrastructure for domain-specific reward functions
- **PPO compatible**: Integrates directly with rLLM training pipeline

### For Agent Development  
- **Tool ecosystem access**: Full Strands tool library (browser, APIs, memory)
- **Robust error handling**: Validation, normalization, intelligent fallbacks
- **Mixed dispatch**: Works with tool-calling and non-tool-calling models

### For Production
- **Memory management**: Proper cleanup and resource handling
- **Scalability**: Ray-based distributed training support
- **Monitoring**: Built-in trajectory verification and debugging

## Advanced Usage

### Custom Reward Functions

```python
def browser_research_reward(step):
    """Reward function for web research tasks"""
    if "error" in str(step.action):
        return -0.1  # Penalize errors
    
    if step.done and "answer" in str(step.action):
        return 1.0   # Reward task completion
    
    if "navigate" in str(step.action):
        return 0.1   # Small reward for exploration
    
    return 0.0       # Neutral for other actions

# Apply rewards after episode
for step in agent.trajectory.steps:
    step.reward = browser_research_reward(step)
```

### Multi-Agent Training

```python
# Train multiple specialized agents
agents = {
    "researcher": StrandsAgent(model, tools=[browser]),
    "analyzer": StrandsAgent(model, tools=[calculator, database]),
    "writer": StrandsAgent(model, tools=[editor, formatter])
}

# Collect trajectories from all agents
trajectories = []
for agent_name, agent in agents.items():
    trajectory = await run_episode(agent, tasks[agent_name])
    trajectories.append((agent_name, trajectory))

# Train with multi-agent PPO
ppo_trainer.train_multi_agent(trajectories)
```

## Troubleshooting

### Common Issues

**Tool_calls not working**: This is often normal LLM behavior. Modern models choose text responses for complex reasoning. The mixed dispatch handles both scenarios.

**Validation errors**: Current implementation uses stricter validation than pure Strands. This catches integration issues early.

**Memory usage**: Ensure proper browser cleanup in finally blocks:

```python
try:
    # Agent execution
    pass
finally:
    browser.browser({"action": {"type": "close", "session_name": "main"}})
```

### Debugging

Enable verbose output:

```bash
export STRANDS_VERBOSE=1
export BROWSER_HEADLESS=false  # See browser actions
```

Check trajectory data:

```python
print(f"Steps recorded: {len(agent.trajectory.steps)}")
for i, step in enumerate(agent.trajectory.steps):
    print(f"Step {i}: obs={step.observation}, action={step.action}")
```

## Conclusion

The rLLM + Strands integration successfully bridges reinforcement learning training with practical agent capabilities. It preserves both frameworks' strengths while enabling RL training of sophisticated web-capable agents.

The key insight is routing all execution through `StrandsAgent.invoke_async()` to maintain RL trajectory collection while leveraging Strands' rich tool ecosystem.

This architecture is ready for production RL training workflows and scales to complex multi-agent scenarios.