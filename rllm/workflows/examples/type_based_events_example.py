"""
Example demonstrating the type-based event system.
This shows how to create custom Event classes and use them in workflows.
"""

import asyncio
from dataclasses import dataclass

from rllm.workflows.event_driven_workflow import Event, EventDrivenWorkflow, EventDrivenWorkflowRunner, StartEvent, StopEvent, step


# Define custom Event types
@dataclass
class UserInputEvent(Event):
    """Event triggered when user provides input"""

    user_message: str = ""

    def __post_init__(self):
        super().__post_init__()
        if not self.user_message and "message" in self.data:
            self.user_message = self.data["message"]


@dataclass
class ProcessingEvent(Event):
    """Event triggered when processing is needed"""

    pass


@dataclass
class ModelResponseEvent(Event):
    """Event triggered when model provides a response"""

    response: str = ""

    def __post_init__(self):
        super().__post_init__()
        if not self.response and "response" in self.data:
            self.response = self.data["response"]


@dataclass
class ValidationEvent(Event):
    """Event triggered when validation is needed"""

    pass


@dataclass
class ErrorEvent(Event):
    """Event triggered when an error occurs"""

    error_message: str = ""

    def __post_init__(self):
        super().__post_init__()
        if not self.error_message and "error" in self.data:
            self.error_message = self.data["error"]


# Mock components
class MockLLMEngine:
    """Mock LLM engine for testing"""

    def __init__(self):
        self.responses = [
            "Hello! How can I help you today?",
            "That's an interesting question. Let me think about it.",
            "Based on your input, here's what I recommend...",
        ]
        self.response_index = 0

    async def get_model_response(self, messages, application_id=None, **kwargs):
        response = self.responses[self.response_index % len(self.responses)]
        self.response_index += 1
        return response


class MockEngine:
    def __init__(self):
        self.rollout_engine = MockLLMEngine()


# Example workflow using type-based events
class ChatWorkflow(EventDrivenWorkflow):
    """Example workflow demonstrating type-based events"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conversation_history = []
        self.max_response_length = 2048

    @step
    async def initialize_chat(self, event: StartEvent) -> UserInputEvent:
        """Initialize the chat workflow"""
        print("🚀 Starting chat workflow...")

        # Simulate initial user message
        initial_message = event.data.get("initial_message", "Hello, how are you?")
        return UserInputEvent(user_message=initial_message, data={"message": initial_message, "turn": 1})

    @step
    async def process_user_input(self, event: UserInputEvent) -> ProcessingEvent:
        """Process user input and prepare for model response"""
        print(f"👤 User: {event.user_message}")

        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": event.user_message})

        return ProcessingEvent(data={"turn": event.data.get("turn", 1)})

    @step
    async def generate_response(self, event: ProcessingEvent) -> ModelResponseEvent:
        """Generate model response"""
        print("🤖 Generating response...")

        # Get model response
        response = await self.get_current_model_response(self.conversation_history, temperature=0.7)

        return ModelResponseEvent(response=response, data={"response": response, "turn": event.data.get("turn", 1)})

    @step
    async def validate_response(self, event: ModelResponseEvent) -> ValidationEvent:
        """Validate the model response"""
        print(f"🤖 Assistant: {event.response}")

        # Add to conversation history
        self.conversation_history.append({"role": "assistant", "content": event.response})

        return ValidationEvent(data={"validated": True, "turn": event.data.get("turn", 1)})

    @step
    async def check_continuation(self, event: ValidationEvent) -> UserInputEvent | StopEvent:
        """Check if conversation should continue"""
        turn = event.data.get("turn", 1)

        if turn >= 3:  # Stop after 3 turns
            print("💬 Conversation completed!")
            return StopEvent(data={"reason": "max_turns_reached", "turns": turn})

        # Simulate next user message
        next_messages = ["That's helpful, can you tell me more?", "Interesting! What about edge cases?", "Thank you for the explanation!"]

        next_message = next_messages[turn - 1] if turn <= len(next_messages) else "Thanks!"

        return UserInputEvent(user_message=next_message, data={"message": next_message, "turn": turn + 1})

    @step
    async def handle_error(self, event: ErrorEvent) -> StopEvent:
        """Handle any errors that occur"""
        print(f"❌ Error occurred: {event.error_message}")
        return StopEvent(data={"reason": "error", "error": event.error_message})


async def run_chat_example():
    """Run the chat workflow example"""
    print("=" * 50)
    print("Type-Based Event System Example")
    print("=" * 50)

    # Create workflow and runner
    workflow = ChatWorkflow()
    runner = EventDrivenWorkflowRunner(workflow)

    # Show registered event handlers
    handlers = runner.get_event_handlers_info()
    print("📋 Registered Event Handlers:")
    for event_type, handler_names in handlers.items():
        print(f"  • {event_type}: {', '.join(handler_names)}")
    print()

    # Run the workflow
    task = {"initial_message": "Hello! I'd like to learn about machine learning."}
    engine = MockEngine()

    try:
        trajectories = await runner.run_workflow("chat_example", task, engine)
        print(f"\n✅ Workflow completed with {len(trajectories)} trajectories")

        # Show event history
        history = runner.get_event_history()
        print(f"\n📚 Event History ({len(history)} events):")
        for i, event in enumerate(history, 1):
            event_type = type(event).__name__
            # Show key data for each event
            key_data = {}
            if hasattr(event, "user_message") and event.user_message:
                key_data["message"] = event.user_message
            elif hasattr(event, "response") and event.response:
                key_data["response"] = event.response[:50] + "..." if len(event.response) > 50 else event.response
            elif hasattr(event, "error_message") and event.error_message:
                key_data["error"] = event.error_message

            print(f"  {i:2d}. {event_type:<20} {key_data}")

    except Exception as e:
        print(f"❌ Error running workflow: {e}")


if __name__ == "__main__":
    asyncio.run(run_chat_example())
