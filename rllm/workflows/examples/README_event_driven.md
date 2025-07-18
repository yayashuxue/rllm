# Event-Driven Workflow System

This document describes the event-driven workflow system that extends the base `Workflow` class to support type-based event execution patterns.

## Overview

The event-driven workflow system allows you to create workflows where:
- Each function is decorated with a `@step` decorator
- Functions are triggered by specific Event types
- Functions can emit events that trigger other functions
- The workflow continues until a `StopEvent` is reached
- Built-in error handling, timeouts, and infinite loop protection
- Type-safe event handling using custom Event classes

## Core Components

### 1. Events

Events are the fundamental communication mechanism in the workflow. All events must inherit from the base `Event` class:

```python
from dataclasses import dataclass
from rllm.workflows.event_driven_workflow import Event, StartEvent, StopEvent

# Define custom Event types
@dataclass
class UserInputEvent(Event):
    """Event triggered when user provides input"""
    user_message: str = ""

@dataclass
class ProcessingEvent(Event):
    """Event triggered when processing is needed"""
    pass

@dataclass
class ValidationEvent(Event):
    """Event triggered when validation is needed"""
    pass

# Create event instances
user_event = UserInputEvent(user_message="Hello", data={"timestamp": "2024-01-01"})
processing_event = ProcessingEvent(data={"task_id": "123"})

# Built-in special events
start_event = StartEvent(data={"task_id": "123", "task": task_data})
stop_event = StopEvent(data={"completed": True, "result": "success"})
```

**Key Changes from Name-Based System:**
- Events no longer have a `name` attribute
- Event identification is based on the actual Python class type
- Users must create custom Event classes by inheriting from `Event`
- Type hints are used for automatic handler registration
- More type-safe and IDE-friendly

### 2. Step Decorator

The `@step` decorator marks functions as event handlers and automatically infers trigger and emit events based on type hints:

```python
from dataclasses import dataclass
from rllm.workflows.event_driven_workflow import step, Event, StartEvent, StopEvent, EventDrivenWorkflow
from typing import Union

# Define custom events first
@dataclass
class InitializedEvent(Event):
    """Event emitted when initialization is complete"""
    pass

@dataclass  
class ProcessingEvent(Event):
    """Event emitted when processing is needed"""
    pass

class MyWorkflow(EventDrivenWorkflow):
    @step
    async def initialize(self, event: StartEvent) -> InitializedEvent:
        """This function is triggered by StartEvent and emits InitializedEvent"""
        # Do initialization work
        return InitializedEvent(data={"ready": True})
    
    @step
    async def process(self, event: InitializedEvent) -> ProcessingEvent:
        """This function is triggered by InitializedEvent and emits ProcessingEvent"""
        # Do processing work
        return ProcessingEvent(data={"processed": True})
    
    @step
    async def complete(self, event: ProcessingEvent) -> StopEvent:
        """This function is triggered by ProcessingEvent and emits StopEvent"""
        return StopEvent(data={"finished": True})
```

**How it works:**
- **Trigger events** are inferred from the function parameter type hints (e.g., `event: StartEvent`)
- **Emit events** are inferred from the return type hints and actual returned events at runtime
- Use `event: Event` for functions that can handle any event type (base Event class matches all events)
- Use specific event types like `event: StartEvent` for functions that only handle specific events
- Use `Union[EventA, EventB]` for functions that handle multiple specific event types
- The system matches events by their Python class type using `isinstance()` checks
- Supports inheritance: handlers for `Event` will also receive subclass events

### 3. EventDrivenWorkflow

The base class for creating event-driven workflows:

```python
class MyEventWorkflow(EventDrivenWorkflow):
    def __init__(self, agent_cls, env_cls, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent_cls()
        self.env = env_cls()
        
    # Add @step decorated methods here
```

### 4. EventDrivenWorkflowRunner

A utility class for running and managing event-driven workflows:

```python
from rllm.workflows.event_driven_workflow import EventDrivenWorkflowRunner

workflow = MyEventWorkflow(agent_cls=MyAgent, env_cls=MyEnv)
runner = EventDrivenWorkflowRunner(workflow)

# Run the workflow
trajectories = await runner.run_workflow("task_1", task_data, engine)

# Get event history
history = runner.get_event_history()

# Get handler information
handlers = runner.get_event_handlers_info()
```

## Usage Patterns

### 1. Simple Linear Workflow

```python
from dataclasses import dataclass

@dataclass
class Step1Event(Event):
    pass

@dataclass
class Step2Event(Event):
    pass

class LinearWorkflow(EventDrivenWorkflow):
    @step
    async def start(self, event: StartEvent) -> Step1Event:
        return Step1Event(data={})
    
    @step
    async def step1(self, event: Step1Event) -> Step2Event:
        return Step2Event(data={})
    
    @step
    async def step2(self, event: Step2Event) -> StopEvent:
        return StopEvent(data={"completed": True})
```

### 2. Conditional Branching

```python
@dataclass
class PathAEvent(Event):
    pass

@dataclass
class PathBEvent(Event):
    pass

class ConditionalWorkflow(EventDrivenWorkflow):
    @step
    async def decide_path(self, event: StartEvent) -> PathAEvent | PathBEvent:
        condition = event.data.get("condition", False)
        if condition:
            return PathAEvent(data={})
        else:
            return PathBEvent(data={})
    
    @step
    async def handle_path_a(self, event: PathAEvent) -> StopEvent:
        return StopEvent(data={"path": "a"})
    
    @step
    async def handle_path_b(self, event: PathBEvent) -> StopEvent:
        return StopEvent(data={"path": "b"})
```

### 3. Loop with Conditions

```python
@dataclass
class ProcessEvent(Event):
    iteration: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        if not self.iteration and "iteration" in self.data:
            self.iteration = self.data["iteration"]

@dataclass
class ContinueLoopEvent(Event):
    pass

class LoopingWorkflow(EventDrivenWorkflow):
    def __init__(self, max_iterations=5, **kwargs):
        super().__init__(**kwargs)
        self.iteration_count = 0
        self.max_iterations = max_iterations
    
    @step
    async def check_continue(self, event: StartEvent | ContinueLoopEvent) -> ProcessEvent | StopEvent:
        self.iteration_count += 1
        
        if self.iteration_count <= self.max_iterations:
            return ProcessEvent(iteration=self.iteration_count, data={"iteration": self.iteration_count})
        else:
            return StopEvent(data={"iterations": self.iteration_count})
    
    @step
    async def do_work(self, event: ProcessEvent) -> ContinueLoopEvent:
        # Do some work
        print(f"Processing iteration {event.iteration}")
        
        return ContinueLoopEvent(data={"iteration": event.iteration})
```

### 4. Error Handling

```python
@dataclass
class WorkDoneEvent(Event):
    result: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        if not self.result and "result" in self.data:
            self.result = self.data["result"]

@dataclass
class ErrorEvent(Event):
    error_message: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        if not self.error_message and "error" in self.data:
            self.error_message = self.data["error"]

class ErrorHandlingWorkflow(EventDrivenWorkflow):
    @step
    async def do_risky_work(self, event: StartEvent) -> WorkDoneEvent | ErrorEvent:
        try:
            # Risky work here
            result = await self.some_risky_operation()
            return WorkDoneEvent(result=result, data={"result": result})
        except Exception as e:
            return ErrorEvent(error_message=str(e), data={"error": str(e)})
    
    @step
    async def handle_success(self, event: WorkDoneEvent) -> StopEvent:
        return StopEvent(data={"success": True, "result": event.result})
    
    @step
    async def handle_error(self, event: ErrorEvent) -> StopEvent:
        print(f"Handling error: {event.error_message}")
        return StopEvent(data={"success": False, "error": event.error_message})
```

## Advanced Features

### 1. Multiple Event Handlers for Same Event

```python
class MultiHandlerWorkflow(EventDrivenWorkflow):
    @step(trigger_events=["data_received"], emit_events=["processed"])
    async def process_data(self, event: Event):
        # Process data
        return Event(name="processed", data={})
    
    @step(trigger_events=["data_received"], emit_events=[])  # No events emitted
    async def log_data_received(self, event: Event):
        # Log the event
        print(f"Data received: {event.data}")
        return None  # No events to emit
```

### 2. Event Data Transformation

```python
class TransformWorkflow(EventDrivenWorkflow):
    @step(trigger_events=[StartEvent], emit_events=["data_ready"])
    async def prepare_data(self, event: Event):
        raw_data = event.data.get("raw_data", [])
        processed_data = [x * 2 for x in raw_data]  # Transform data
        
        return Event(name="data_ready", data={"processed": processed_data})
    
    @step(trigger_events=["data_ready"], emit_events=[StopEvent])
    async def use_data(self, event: Event):
        processed = event.data.get("processed", [])
        result = sum(processed)
        
        return StopEvent(data={"final_result": result})
```

### 3. Agent-Environment Integration

```python
class AgentEnvWorkflow(EventDrivenWorkflow):
    def __init__(self, agent_cls, env_cls, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent_cls()
        self.env = env_cls()
    
    @step(trigger_events=[StartEvent], emit_events=["env_ready"])
    async def initialize_environment(self, event: Event):
        task = event.data.get("task")
        observation, info = await run_in_executor(
            self.current_engine.executor, 
            self.env.reset, 
            task
        )
        self.agent.reset(task)
        self.agent.update_from_env(observation, 0, False, info)
        
        return Event(name="env_ready", data={"observation": observation})
    
    @step(trigger_events=["env_ready", "action_executed"], emit_events=["action_generated"])
    async def generate_action(self, event: Event):
        prompt = self.agent.chat_completions
        response = await self.get_current_model_response(prompt)
        action = self.agent.update_from_model(response)
        
        return Event(name="action_generated", data={"action": action})
    
    @step(trigger_events=["action_generated"], emit_events=["action_executed", StopEvent])
    async def execute_action(self, event: Event):
        action = event.data.get("action")
        
        next_obs, reward, done, info = await run_in_executor(
            self.current_engine.executor, 
            self.env.step, 
            action
        )
        self.agent.update_from_env(next_obs, reward, done, info)
        
        if done:
            self.trajectories.append(self.agent.trajectory)
            return StopEvent(data={"completed": True, "reward": reward})
        else:
            return Event(name="action_executed", data={"reward": reward})
```

## Configuration Options

The `EventDrivenWorkflow` constructor accepts several configuration options:

```python
workflow = EventDrivenWorkflow(
    max_events=1000,        # Maximum events to prevent infinite loops
    event_timeout=30.0,     # Timeout in seconds for event processing
    enforce_max_prompt_length=True,  # From base Workflow class
    accumulate_response_length=True, # From base Workflow class
)
```

## Testing and Debugging

### 1. Event History

```python
runner = EventDrivenWorkflowRunner(workflow)
trajectories = await runner.run_workflow("task_1", task_data, engine)

# Get event history for debugging
history = runner.get_event_history()
for i, event in enumerate(history):
    print(f"{i+1}. {event.name} at {event.timestamp}: {event.data}")
```

### 2. Handler Information

```python
handlers = runner.get_event_handlers_info()
print("Event handlers:")
for event_name, handler_names in handlers.items():
    print(f"  {event_name}: {handler_names}")
```

### 3. Manual Event Triggering

```python
# For testing specific event paths
await runner.trigger_manual_event("custom_event", {"test": True})
```

## Best Practices

1. **Keep Steps Small**: Each step should do one specific thing
2. **Use Descriptive Event Names**: Make event names clear and specific
3. **Handle Errors**: Always include error handling steps
4. **Add Logging**: Use steps that don't emit events for logging/monitoring
5. **Test Edge Cases**: Test timeout, error conditions, and edge cases
6. **Document Event Flow**: Comment the expected event flow in your workflow

## Migration from Traditional Workflows

To convert existing workflows to event-driven:

1. Identify the main steps in your workflow
2. Convert each step to a `@step` decorated method
3. Define the event flow between steps
4. Add error handling and completion logic
5. Test with the `EventDrivenWorkflowRunner`

Example migration:

```python
# Traditional workflow
class OldWorkflow(Workflow):
    async def __call__(self, task_id, task, engine):
        # Step 1
        obs, info = self.env.reset(task)
        # Step 2
        action = self.agent.act(obs)
        # Step 3
        next_obs, reward, done, info = self.env.step(action)
        return trajectories

# Event-driven workflow
class NewWorkflow(EventDrivenWorkflow):
    @step(trigger_events=[StartEvent], emit_events=["obs_ready"])
    async def reset_env(self, event: Event):
        # Step 1
        
    @step(trigger_events=["obs_ready"], emit_events=["action_ready"])  
    async def generate_action(self, event: Event):
        # Step 2
        
    @step(trigger_events=["action_ready"], emit_events=[StopEvent])
    async def execute_action(self, event: Event):
        # Step 3
```

This event-driven approach provides more flexibility, better error handling, and clearer workflow logic compared to traditional linear workflows. 