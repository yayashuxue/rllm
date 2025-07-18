import asyncio
import inspect
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Union, get_args, get_origin, get_type_hints

from rllm.agents.agent import Trajectory
from rllm.workflows.workflow import Workflow


@dataclass
class Event:
    """Base event class for event-driven workflows"""

    data: dict[str, Any] = field(default_factory=dict)
    source: str | None = None
    timestamp: float | None = None

    def __post_init__(self):
        if self.timestamp is None:
            import time

            self.timestamp = time.time()


@dataclass
class StopEvent(Event):
    """Special event that signals workflow termination"""

    pass


@dataclass
class StartEvent(Event):
    """Special event that signals workflow start"""

    pass


class EventHandler:
    """Container for event handler metadata"""

    def __init__(self, func: Callable):
        self.func = func
        self.trigger_events = self._parse_trigger_events(func)
        self.emit_events = self._parse_emit_events(func)

    def _parse_trigger_events(self, func: Callable) -> list[type]:
        """Parse trigger events from function signature type hints"""
        trigger_events = []

        try:
            # Get type hints for the function
            type_hints = get_type_hints(func)
            sig = inspect.signature(func)

            for param_name, param in sig.parameters.items():
                # Skip 'self' parameter
                if param_name == "self":
                    continue

                # Get type hint for this parameter
                param_type = type_hints.get(param_name)
                if param_type and self._is_event_type(param_type):
                    trigger_events.append(param_type)

        except (NameError, AttributeError, TypeError):
            # If type hints are not available or malformed, we'll need runtime detection
            pass

        # If no trigger events found, assume it can be triggered by any Event
        if not trigger_events:
            trigger_events = [Event]  # Base Event type as wildcard

        return trigger_events

    def _parse_emit_events(self, func: Callable) -> list[type]:
        """Parse emit events from function return type hint"""
        emit_events = []

        try:
            # Get return type hint
            type_hints = get_type_hints(func)
            return_type = type_hints.get("return")

            if return_type:
                # Handle Union types (e.g., Union[Event, StopEvent])
                if get_origin(return_type) is Union:
                    union_args = get_args(return_type)
                    for arg_type in union_args:
                        if self._is_event_type(arg_type):
                            emit_events.append(arg_type)
                else:
                    # Handle single return type
                    if self._is_event_type(return_type):
                        emit_events.append(return_type)

        except (NameError, AttributeError, TypeError):
            # If return type hint is not available, we'll determine emit events at runtime
            pass

        return emit_events

    def _is_event_type(self, event_type: type) -> bool:
        """Check if a type is an Event type"""
        if event_type is None:
            return False

        # Handle direct Event classes
        if inspect.isclass(event_type) and issubclass(event_type, Event):
            return True

        return False


def step(func: Callable = None):
    """
    Decorator to mark a function as an event-driven step.

    Automatically infers:
    - Trigger events from function parameter type hints
    - Emit events from function return type hints and runtime returns

    Usage:
        @step
        async def my_handler(self, event: StartEvent) -> CustomEvent:
            return CustomEvent(data={})
    """

    def decorator(f: Callable):
        # Store event handler metadata on the function
        f._event_handler = EventHandler(f)

        @wraps(f)
        async def wrapper(*args, **kwargs):
            result = await f(*args, **kwargs)

            # Track actual emit events at runtime for better accuracy
            if hasattr(f, "_event_handler") and result:
                runtime_events = []
                if isinstance(result, list | tuple):
                    for item in result:
                        if isinstance(item, Event):
                            event_type = type(item)
                            if event_type not in runtime_events:
                                runtime_events.append(event_type)
                elif isinstance(result, Event):
                    event_type = type(result)
                    runtime_events.append(event_type)

                # Update emit events with runtime information
                if runtime_events:
                    # Combine static and runtime emit events
                    all_emit_events = list(set(f._event_handler.emit_events + runtime_events))
                    f._event_handler.emit_events = all_emit_events

            return result

        # Preserve the event handler metadata on the wrapper
        wrapper._event_handler = f._event_handler
        return wrapper

    # Support both @step and @step() syntax
    if func is None:
        return decorator
    else:
        return decorator(func)


class EventDrivenWorkflow(Workflow):
    """
    Event-driven workflow that executes steps based on events.
    Each step is decorated with @step and specifies which events it handles.
    """

    def __init__(self, max_events: int = 1000, event_timeout: float = 30.0, **kwargs):
        super().__init__(**kwargs)
        self.max_events = max_events
        self.event_timeout = event_timeout
        self.event_queue = deque()
        self.event_handlers = defaultdict(list)  # event_type -> list of handlers
        self.event_history = []
        self.current_task = None
        self.current_engine = None
        self.trajectories = []

        # Register all step methods
        self._register_event_handlers()

    def _find_matching_handlers(self, event: Event) -> list:
        """Find handlers that can process the given event"""
        matching_handlers = []

        # Exact type match
        event_type = type(event)
        if event_type in self.event_handlers:
            matching_handlers.extend(self.event_handlers[event_type])

        # Check for handlers that can handle parent types
        for handler_type, handlers in self.event_handlers.items():
            if handler_type != event_type and isinstance(event, handler_type):
                matching_handlers.extend(handlers)

        return matching_handlers

    def _register_event_handlers(self):
        """Discover and register all methods decorated with @step"""
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue

            attr = getattr(self, attr_name)
            if hasattr(attr, "_event_handler"):
                handler = attr._event_handler
                for event_type in handler.trigger_events:
                    self.event_handlers[event_type].append(handler)

    async def emit_event(self, event: Event) -> None:
        """Emit an event to the workflow"""
        if not isinstance(event, Event):
            raise TypeError("event must be an instance of Event or its subclasses")

        self.event_queue.append(event)
        self.event_history.append(event)

        # Prevent infinite loops
        if len(self.event_history) > self.max_events:
            raise RuntimeError(f"Maximum number of events ({self.max_events}) exceeded")

    async def process_events(self) -> None:
        """Process all events in the queue until StopEvent is reached"""
        start_time = asyncio.get_event_loop().time()

        while self.event_queue:
            # Check for timeout
            if asyncio.get_event_loop().time() - start_time > self.event_timeout:
                raise TimeoutError(f"Event processing timeout ({self.event_timeout}s) exceeded")

            event = self.event_queue.popleft()

            # Check for stop event
            if isinstance(event, StopEvent):
                break

            # Find and execute handlers for this event
            handlers = self._find_matching_handlers(event)

            for handler in handlers:
                try:
                    # Execute the handler
                    result = await handler.func(self, event)

                    # If the handler returns events, emit them
                    if result:
                        if isinstance(result, list | tuple):
                            for emit_event in result:
                                await self.emit_event(emit_event)
                        else:
                            await self.emit_event(result)

                except Exception as e:
                    print(f"Error in event handler {handler.func.__name__} for event {type(event).__name__}: {e}")
                    # Create a custom error event if one doesn't exist
                    # Users should define their own ErrorEvent class
                    error_event = Event(data={"error": str(e), "handler": handler.func.__name__})
                    await self.emit_event(error_event)

    async def __call__(self, task_id: str, task: dict, engine, **kwargs) -> list[Trajectory]:
        """Execute the event-driven workflow"""
        self.current_task = task
        self.current_engine = engine
        self.trajectories = []
        self.event_queue.clear()
        self.event_history.clear()

        # Reset all components
        self.reset(task)

        # Start the workflow by emitting a start event
        await self.emit_event(StartEvent(data={"task_id": task_id, "task": task}))

        # Process events until completion
        await self.process_events()

        return self.process_and_collect_trajectories()

    def _collect_trajectories(self) -> list[Trajectory]:
        """Collect trajectories from the workflow"""
        return self.trajectories

    async def get_current_model_response(self, messages: list[dict], **kwargs) -> str:
        """Helper method to get model response using current engine"""
        if not self.current_engine:
            raise RuntimeError("No current engine available")

        task_id = getattr(self.current_task, "id", "unknown") if self.current_task else "unknown"
        return await self.get_model_response(self.current_engine.rollout_engine, messages, task_id, **kwargs)


class EventDrivenWorkflowRunner:
    """
    Runner for event-driven workflows that provides utilities for workflow execution.
    """

    def __init__(self, workflow: EventDrivenWorkflow):
        self.workflow = workflow

    async def run_workflow(self, task_id: str, task: dict, engine, **kwargs) -> list[Trajectory]:
        """Run the workflow and return trajectories"""
        return await self.workflow(task_id, task, engine, **kwargs)

    def get_event_history(self) -> list[Event]:
        """Get the history of events processed in the last workflow run"""
        return self.workflow.event_history

    def get_event_handlers_info(self) -> dict[str, list[str]]:
        """Get information about registered event handlers"""
        info = {}
        for event_type, handlers in self.workflow.event_handlers.items():
            type_name = event_type.__name__ if hasattr(event_type, "__name__") else str(event_type)
            info[type_name] = [handler.func.__name__ for handler in handlers]
        return info

    async def trigger_manual_event(self, event: Event) -> None:
        """Manually trigger an event (useful for testing or external triggers)"""
        await self.workflow.emit_event(event)
        await self.workflow.process_events()
