"""
Test file demonstrating the event-driven workflow system.
This file shows how to create and run event-driven workflows with mock components.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from rllm.workflows.event_driven_workflow import Event, EventDrivenWorkflow, EventDrivenWorkflowRunner, StartEvent, StopEvent, step


# Custom Event types for testing
@dataclass
class InitializedEvent(Event):
    """Event emitted when initialization is complete"""

    pass


@dataclass
class ActionNeededEvent(Event):
    """Event emitted when an action is needed"""

    pass


@dataclass
class ActionExecutedEvent(Event):
    """Event emitted when an action has been executed"""

    pass


@dataclass
class DecisionMadeEvent(Event):
    """Event emitted when a decision has been made"""

    pass


@dataclass
class ReadyToDecideEvent(Event):
    """Event emitted when ready to make a decision"""

    pass


@dataclass
class NeedHelpEvent(Event):
    """Event emitted when help is needed"""

    pass


@dataclass
class RetryDecisionEvent(Event):
    """Event emitted when a decision should be retried"""

    pass


@dataclass
class ErrorEvent(Event):
    """Event emitted when an error occurs"""

    pass


# Mock components for testing
@dataclass
class MockStep:
    """Mock step for trajectory"""

    observation: str
    action: str
    reward: float
    done: bool
    info: dict[str, Any]
    chat_completions: list[dict[str, str]]


class MockTrajectory:
    """Mock trajectory for testing"""

    def __init__(self):
        self.steps = []
        self.reward = 0.0
        self.mc_return = 0.0

    def add_step(self, observation, action, reward, done, info, chat_completions):
        step = MockStep(observation, action, reward, done, info, chat_completions)
        self.steps.append(step)


class MockAgent:
    """Mock agent for testing"""

    def __init__(self, **kwargs):
        self.trajectory = MockTrajectory()
        self.chat_completions = []
        self.current_observation = ""
        self.current_action = ""

    def reset(self, task=None):
        self.trajectory = MockTrajectory()
        self.chat_completions = []
        self.current_observation = ""
        self.current_action = ""

    def update_from_env(self, observation, reward, done, info):
        self.current_observation = observation
        # Add environment observation to chat completions
        self.chat_completions.append({"role": "user", "content": f"Observation: {observation}"})

    def update_from_model(self, response):
        # Add model response to chat completions
        self.chat_completions.append({"role": "assistant", "content": response})
        self.current_action = response

        # Add step to trajectory
        self.trajectory.add_step(
            observation=self.current_observation,
            action=response,
            reward=0.0,  # Will be updated later
            done=False,
            info={},
            chat_completions=self.chat_completions.copy(),
        )

        return response


class MockEnvironment:
    """Mock environment for testing"""

    def __init__(self, **kwargs):
        self.step_count = 0
        self.max_steps = 3
        self.reset_called = False

    def reset(self, task=None):
        self.step_count = 0
        self.reset_called = True
        return f"Initial observation for task: {task.get('name', 'unknown')}", {"reset": True}

    def step(self, action):
        self.step_count += 1
        observation = f"Step {self.step_count} observation after action: {action}"
        reward = 1.0 if "good" in action.lower() else 0.5
        done = self.step_count >= self.max_steps
        info = {"step": self.step_count}
        return observation, reward, done, info


class MockRolloutEngine:
    """Mock rollout engine for testing"""

    def __init__(self):
        self.responses = ["I will take a good action", "Let me do something good", "This is a good final action"]
        self.response_index = 0

    async def get_model_response(self, messages, application_id=None, max_tokens=None, **kwargs):
        response = self.responses[self.response_index % len(self.responses)]
        self.response_index += 1
        return response


class MockEngine:
    """Mock engine for testing"""

    def __init__(self):
        self.executor = None  # We'll use direct calls instead of executor
        self.rollout_engine = MockRolloutEngine()


# Test workflow implementations
class SimpleTestWorkflow(EventDrivenWorkflow):
    """Simple test workflow for basic functionality"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent = MockAgent()
        self.env = MockEnvironment()
        self.test_state = {"initialized": False, "steps": 0}
        # Add required attributes for base workflow
        self.max_response_length = 2048

    @step
    async def initialize(self, event: StartEvent) -> InitializedEvent:
        """Initialize the workflow"""
        print("Initializing simple test workflow")
        task = event.data.get("task", {})

        # Reset environment and agent
        observation, info = self.env.reset(task)
        self.agent.reset(task)
        self.agent.update_from_env(observation, 0, False, info)

        self.test_state["initialized"] = True

        return InitializedEvent(data={"observation": observation})

    @step
    async def start_interaction(self, event: InitializedEvent) -> ActionNeededEvent:
        """Start the interaction loop"""
        return ActionNeededEvent(data={"step": 0})

    @step
    async def take_action(self, event: ActionNeededEvent) -> ActionNeededEvent | StopEvent:
        """Take an action and check if done"""
        step_num = event.data.get("step", 0)

        # Get model response
        response = await self.get_current_model_response(self.agent.chat_completions, temperature=0.7)

        # Update agent with model response
        action = self.agent.update_from_model(response)

        # Execute action in environment
        observation, reward, done, info = self.env.step(action)
        self.agent.update_from_env(observation, reward, done, info)

        self.test_state["steps"] += 1

        if done:
            # Add final trajectory
            self.trajectories.append(self.agent.trajectory)
            return StopEvent(data={"completed": True, "steps": self.test_state["steps"]})
        else:
            return ActionNeededEvent(data={"step": step_num + 1})


class ConditionalTestWorkflow(EventDrivenWorkflow):
    """Test workflow with conditional logic and multiple event types"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent = MockAgent()
        self.env = MockEnvironment()
        self.decision_count = 0
        self.error_count = 0
        # Add required attributes for base workflow
        self.max_response_length = 2048

    @step
    async def setup(self, event: StartEvent) -> ReadyToDecideEvent:
        """Setup the workflow"""
        task = event.data.get("task", {})
        observation, info = self.env.reset(task)
        self.agent.reset(task)
        self.agent.update_from_env(observation, 0, False, info)

        return ReadyToDecideEvent(data={"setup_complete": True})

    @step
    async def make_decision(self, event: ReadyToDecideEvent) -> DecisionMadeEvent | NeedHelpEvent | ErrorEvent:
        """Make a decision based on current state"""
        self.decision_count += 1

        # Simulate different decision outcomes
        if self.decision_count == 1:
            return DecisionMadeEvent(data={"decision": "explore", "confidence": 0.8})
        elif self.decision_count == 2:
            return NeedHelpEvent(data={"reason": "uncertain"})
        elif self.decision_count == 3:
            return ErrorEvent(data={"message": "Simulated error"})
        else:
            return DecisionMadeEvent(data={"decision": "complete", "confidence": 1.0})

    @step
    async def execute_decision(self, event: DecisionMadeEvent) -> ActionExecutedEvent | StopEvent:
        """Execute the decision"""
        decision = event.data.get("decision")

        if decision == "complete":
            self.trajectories.append(self.agent.trajectory)
            return StopEvent(data={"reason": "completed", "decisions": self.decision_count})

        # Simulate action execution
        response = await self.get_current_model_response(self.agent.chat_completions, temperature=0.5)

        action = self.agent.update_from_model(response)
        observation, reward, done, info = self.env.step(action)
        self.agent.update_from_env(observation, reward, done, info)

        return ActionExecutedEvent(data={"action": action, "reward": reward})

    @step
    async def continue_loop(self, event: ActionExecutedEvent) -> ReadyToDecideEvent:
        """Continue the decision loop"""
        return ReadyToDecideEvent(data={"continuing": True})

    @step
    async def handle_help_request(self, event: NeedHelpEvent) -> RetryDecisionEvent:
        """Handle help requests"""
        print(f"Handling help request: {event.data.get('reason')}")
        return RetryDecisionEvent(data={"help_provided": True})

    @step
    async def handle_error(self, event: ErrorEvent) -> RetryDecisionEvent | StopEvent:
        """Handle errors"""
        self.error_count += 1
        error_msg = event.data.get("message", "Unknown error")
        print(f"Handling error ({self.error_count}): {error_msg}")

        if self.error_count >= 2:
            return StopEvent(data={"reason": "too_many_errors", "error_count": self.error_count})
        else:
            return RetryDecisionEvent(data={"error_handled": True})

    @step
    async def retry_decision(self, event: RetryDecisionEvent) -> ReadyToDecideEvent:
        """Handle retry decision events"""
        return ReadyToDecideEvent(data={"retrying": True})


# Test functions
async def test_simple_workflow():
    """Test the simple workflow"""
    print("\n=== Testing Simple Workflow ===")

    workflow = SimpleTestWorkflow(max_events=50)  # Reduce to see what's happening
    runner = EventDrivenWorkflowRunner(workflow)

    # Show event handlers
    handlers = runner.get_event_handlers_info()
    print("Event handlers:", handlers)

    # Run workflow
    task = {"name": "test_task", "id": "task_1"}
    engine = MockEngine()

    trajectories = await runner.run_workflow("task_1", task, engine)

    print(f"Workflow completed with {len(trajectories)} trajectories")
    print(f"Final state: {workflow.test_state}")

    # Show event history
    history = runner.get_event_history()
    print(f"Event history ({len(history)} events):")
    for i, event in enumerate(history):
        print(f"  {i + 1}. {type(event).__name__}: {event.data}")


async def test_conditional_workflow():
    """Test the conditional workflow"""
    print("\n=== Testing Conditional Workflow ===")

    workflow = ConditionalTestWorkflow()
    runner = EventDrivenWorkflowRunner(workflow)

    # Run workflow
    task = {"name": "conditional_test", "id": "task_2"}
    engine = MockEngine()

    trajectories = await runner.run_workflow("task_2", task, engine)

    print(f"Workflow completed with {len(trajectories)} trajectories")
    print(f"Decisions made: {workflow.decision_count}")
    print(f"Errors handled: {workflow.error_count}")

    # Show event history
    history = runner.get_event_history()
    print(f"Event history ({len(history)} events):")
    for i, event in enumerate(history):
        print(f"  {i + 1}. {type(event).__name__}: {event.data}")


async def test_manual_event_triggering():
    """Test manual event triggering"""
    print("\n=== Testing Manual Event Triggering ===")

    workflow = SimpleTestWorkflow()
    EventDrivenWorkflowRunner(workflow)

    # Manually trigger events
    print("Triggering manual events...")

    # This would require the workflow to be in a state where it can process events
    # For demo purposes, we'll just show the API
    print("Manual event triggering API available via runner.trigger_manual_event()")


async def test_event_timeout():
    """Test event timeout handling"""
    print("\n=== Testing Event Timeout ===")

    @dataclass
    class SlowProcessEvent(Event):
        """Event for slow processing"""

        pass

    class TimeoutTestWorkflow(EventDrivenWorkflow):
        def __init__(self, **kwargs):
            super().__init__(event_timeout=1.0, **kwargs)  # 1 second timeout

        @step
        async def start_slow_process(self, event: StartEvent) -> SlowProcessEvent:
            return SlowProcessEvent(data={})

        @step
        async def infinite_loop(self, event: SlowProcessEvent) -> SlowProcessEvent:
            await asyncio.sleep(0.1)  # Simulate some work
            return SlowProcessEvent(data={})  # Creates infinite loop

    workflow = TimeoutTestWorkflow()
    runner = EventDrivenWorkflowRunner(workflow)

    try:
        task = {"name": "timeout_test"}
        engine = MockEngine()
        trajectories = await runner.run_workflow("timeout_test", task, engine)
        print(f"Workflow completed with {len(trajectories)} trajectories")
        print("Unexpected: workflow completed without timeout")
    except TimeoutError as e:
        print(f"Expected timeout error: {e}")


async def run_all_tests():
    """Run all tests"""
    print("Starting Event-Driven Workflow Tests")
    print("=" * 50)

    await test_simple_workflow()
    await test_conditional_workflow()
    await test_manual_event_triggering()
    await test_event_timeout()

    print("\n" + "=" * 50)
    print("All tests completed!")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
