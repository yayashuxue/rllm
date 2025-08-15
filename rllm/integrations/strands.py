from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any, TypeVar

from pydantic import BaseModel
from strands import Agent
from strands.models.model import Model
from strands.types.content import ContentBlock, Messages
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolSpec

from rllm.agents.agent import Step, Trajectory
from rllm.engine.rollout_engine import RolloutEngine

T = TypeVar("T", bound=BaseModel)


class RLLMModel(Model):
    """Model class that uses rLLM's RolloutEngine for inference."""
    
    def __init__(self, rollout_engine: RolloutEngine, model_id: str = "gpt-4",  **model_config):
        """Initialize the RLLMModel.
        
        Args:
            rollout_engine: The rLLM RolloutEngine instance to use for inference
            model_id: The model ID to use 
            **model_config: Additional model configuration
        """ 
        self.rollout_engine = rollout_engine
        self.config = {
            "model_id": model_id,
            "params": model_config
        }
        # Optional default tool specs to use when the caller (Agent) does not pass any
        self._default_tool_specs: list[ToolSpec] | None = None

    def update_config(self, **model_config: Any) -> None:
        """Update the model configuration.
        
        Args:
            **model_config: Configuration overrides.
        """
        if "model_id" in model_config:
            self.config["model_id"] = model_config.pop("model_id")
        
        if "params" not in self.config:
            self.config["params"] = {}
        self.config["params"].update(model_config)
    
    def set_default_tool_specs(self, tool_specs: list[ToolSpec] | None) -> None:
        """Set default tool specs used when the caller omits tool_specs.
        
        Args:
            tool_specs: Default tool specifications to use
        """
        self._default_tool_specs = tool_specs
    
    def get_config(self) -> dict[str, Any]:
        """Get the model configuration.
        
        Returns:
            The model's configuration.
        """
        return self.config.copy()
    
    async def structured_output(
        self, 
        output_model: type[T], 
        prompt: Messages, 
        system_prompt: str | None = None, 
        **kwargs: Any
    ) -> AsyncGenerator[dict[str, T | Any], None]:
        """Get structured output from the model.
        
        Note: This is a basic implementation that converts the text response to the output model.
        For more advanced structured output, consider using the native OpenAI structured output features.
        
        Args:
            output_model: The output model to use for the agent.
            prompt: The prompt messages to use for the agent.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments.
            
        Yields:
            Model events with the last being the structured output.
        """
        # Convert Strands messages to chat completion format
        messages = self._convert_messages_to_chat_format(prompt, system_prompt)
        
        # Add instruction for structured output
        if messages and messages[-1]["role"] == "user":
            original_content = messages[-1]["content"]
            messages[-1]["content"] = f"{original_content}\n\nPlease respond with a JSON object that matches this schema: {output_model.model_json_schema()}"
        
        # Get response from rollout engine
        response_text = await self.rollout_engine.get_model_response(
            messages, 
            model=self.config["model_id"],
            **self.config.get("params", {}),
            **kwargs
        )
        
        try:
            # Try to parse the response as JSON and convert to the output model
            import json
            # Extract JSON from response if it's wrapped in other text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                parsed_data = json.loads(json_str)
                structured_output = output_model(**parsed_data)
                yield {"output": structured_output}
            else:
                raise ValueError("No valid JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse structured output: {e}")
    
    def _to_openai_tools(self, tool_specs: list[ToolSpec] | None) -> list[dict[str, Any]]:
        """Convert Strands ToolSpec list to OpenAI chat tools parameter.

        Best-effort mapping: name, description, parameters/schema.
        """
        tools_param: list[dict[str, Any]] = []
        if not tool_specs:
            return tools_param
        for index, spec in enumerate(tool_specs):
            try:
                # Prefer spec from Strands ToolFunc/AgentTool
                ts_dict = getattr(spec, "tool_spec", None)
                if ts_dict is None and isinstance(spec, dict) and isinstance(spec.get("toolSpec"), dict):
                    ts_dict = spec.get("toolSpec")
                # Name
                name = (
                    (ts_dict.get("name") if isinstance(ts_dict, dict) else None)
                    or getattr(spec, "name", None)
                    or getattr(spec, "tool_name", None)
                    or getattr(spec, "__name__", None)
                    or "tool"
                )
                # Description
                description = (
                    (ts_dict.get("description") if isinstance(ts_dict, dict) else None)
                    or getattr(spec, "description", "")
                    or getattr(spec, "desc", "")
                )
                # Schema
                schema = None
                if isinstance(ts_dict, dict):
                    schema = ts_dict.get("inputSchema") or ts_dict.get("parameters") or ts_dict.get("schema")
                schema = schema or getattr(spec, "parameters", None) or getattr(spec, "input_schema", None)
                if schema is None and hasattr(spec, "json"):
                    js = spec.json
                    if isinstance(js, dict):
                        schema = js.get("parameters") or js.get("schema") or js.get("inputSchema")
                if schema is None and hasattr(spec, "model_json_schema"):
                    try:
                        schema = spec.model_json_schema()
                    except Exception:
                        schema = None
                if schema is None:
                    schema = {"type": "object", "properties": {}}
                tools_param.append({
                    "type": "function",
                    "function": {
                        "name": str(name),
                        "description": str(description),
                        "parameters": schema,
                    },
                })
            except Exception as error:
                print(
                    f"[RLLMModel._to_openai_tools] failed to map tool spec at index {index}: {error!r}"
                )
                continue
        return tools_param

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        """Stream conversation with the model using RolloutEngine.
        
        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments.
            
        Yields:
            Formatted message chunks from the model.
        """
        # Convert Strands messages to chat completion format; rely on Strands' own prompting
        chat_messages = self._convert_messages_to_chat_format(messages, system_prompt or "")
        
        # Yield message start
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        
        # Get response from rollout engine
        # Prefer explicitly provided tool specs; fall back to defaults, if any
        effective_tool_specs = tool_specs if tool_specs is not None else self._default_tool_specs
        openai_tools = self._to_openai_tools(effective_tool_specs)
        response_text = await self.rollout_engine.get_model_response(
            chat_messages,
            model=self.config["model_id"],
            tools=openai_tools if openai_tools else None,
            tool_choice="auto" if openai_tools else None,
            **self.config.get("params", {}),
            **kwargs,
        )
        
        # Simulate streaming by yielding the response in chunks
        # In a real streaming implementation, you'd want to modify RolloutEngine to support streaming
        chunk_size = 50  # Adjust as needed
        for i in range(0, len(response_text), chunk_size):
            chunk = response_text[i:i + chunk_size]
            yield {"contentBlockDelta": {"delta": {"text": chunk}}}
        
        # Yield message end
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        
        # TODO: Add usage metadata if available from rollout engine
        yield {
            "metadata": {
                "usage": {
                    "inputTokens": 0,  # Would need to calculate or get from rollout engine
                    "outputTokens": 0,  # Would need to calculate or get from rollout engine  
                    "totalTokens": 0,
                },
                "metrics": {
                    "latencyMs": 0,
                },
            }
        }
    
    def _convert_messages_to_chat_format(
        self, 
        messages: Messages, 
        system_prompt: str | None = None
    ) -> list[dict[str, str]]:
        """Convert Strands messages to chat completion format.
        
        This reuses logic similar to OpenAIModel but outputs the simpler format expected by RolloutEngine.
        
        Args:
            messages: Strands messages to convert
            system_prompt: Optional system prompt to prepend
            
        Returns:
            List of chat completion messages
        """
        chat_messages = []
        
        # Add system prompt if provided
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            # Extract text content from Strands format
            text_content = ""
            for content_block in content:
                if "text" in content_block:
                    text_content += content_block["text"]
                elif "toolUse" in content_block:
                    # For now, represent tool use as text
                    tool_use = content_block["toolUse"]
                    text_content += f"[Tool: {tool_use['name']} with input: {tool_use.get('input', {})}]"
                elif "toolResult" in content_block:
                    # For now, represent tool result as text
                    tool_result = content_block["toolResult"]
                    text_content += f"[Tool Result: {tool_result.get('content', [])}]"
                # TODO: Handle other content types like images, documents if needed
            
            if text_content.strip():  # Only add if there's actual content
                chat_messages.append({"role": role, "content": text_content})
        
        # print("[***_convert_messages_to_chat_format***]", chat_messages)
        return chat_messages


class StrandsAgent(Agent):
    def __init__(self, model: str, **kwargs):
        """Initialize StrandsAgent with trajectory tracking.
        
        Args:
            model: The model to use (can be a string or Model instance)
            **kwargs: Additional arguments to pass to the base Agent class
        """
        # Capture tools argument before base class potentially wraps/moves it
        _init_tools = kwargs.get("tools")

        super().__init__(model=model, **kwargs)
        self._trajectory = Trajectory()
        self._current_step = None

        # Auto-inject default tool specs into RLLMModel for downstream tool-aware backends
        try:
            if isinstance(self.model, RLLMModel) and _init_tools:
                self.model.set_default_tool_specs(_init_tools)
        except Exception:
            pass
        
    @property
    def trajectory(self) -> Trajectory:
        """Get the current trajectory object."""
        return self._trajectory
    
    @property 
    def chat_completions(self):
        """Convert agent's messages into chat completions format."""
        completions = []
        for message in self.messages:
            # Convert Strands message format to chat completion format
            if isinstance(message.get('content'), list):
                # Handle multi-content messages
                text_content = ""
                for content_block in message['content']:
                    if isinstance(content_block, dict) and 'text' in content_block:
                        text_content += content_block['text']
                completions.append({
                    "role": message['role'],
                    "content": text_content
                })
            else:
                # Handle simple string content
                completions.append({
                    "role": message['role'], 
                    "content": str(message.get('content', ''))
                })
        return completions
    
    def reset_trajectory(self, task: Any = None):
        """Reset the trajectory for a new episode."""
        self._trajectory = Trajectory(task=task)
        self._current_step = None
        
    def _start_new_step(self, observation: Any = None):
        """Start a new step in the trajectory."""
        self._current_step = Step(
            chat_completions=self.chat_completions.copy(),
            observation=observation
        )
        
    def _finish_current_step(self, model_response: str = "", action: Any = None, reward: float = 0.0, done: bool = False):
        """Finish the current step and add it to the trajectory."""
        if self._current_step is not None:
            self._current_step.model_response = model_response
            self._current_step.action = action
            self._current_step.reward = reward
            self._current_step.done = done
            self._current_step.chat_completions = self.chat_completions.copy()
            
            self._trajectory.steps.append(self._current_step)
            self._trajectory.reward += reward
            self._current_step = None

    def __call__(self, prompt: str | list[ContentBlock], **kwargs) -> Any:
        """Enhanced call method that tracks trajectory."""
        # Start a new step with the user prompt as observation
        self._start_new_step(observation=prompt)
        
        # Call the original Strands Agent logic
        result = super().__call__(prompt, **kwargs)
        
        # Extract relevant information from the result
        model_response = ""
        action = None
        
        if hasattr(result, 'message') and result.message:
            # Extract text content from the final message
            if hasattr(result.message, 'content') and isinstance(result.message.content, list):
                for content_block in result.message.content:
                    if isinstance(content_block, dict) and 'text' in content_block:
                        model_response += content_block['text']
            elif hasattr(result.message, 'content'):
                model_response = str(result.message.content)
                
            # The action could be the final message or result itself
            action = result.message if hasattr(result, 'message') else result
        
        # Determine if this step is done (could be based on stop_reason or other criteria)
        done = hasattr(result, 'stop_reason') and result.stop_reason in ['end_turn', 'stop_sequence']
        
        # Finish the current step
        self._finish_current_step(
            model_response=model_response,
            action=action,
            done=done
        )
        
        return result
    
    async def invoke_async(self, prompt: str | list[ContentBlock], **kwargs) -> Any:
        """Enhanced async invoke method that tracks trajectory."""
        # Start a new step with the user prompt as observation  
        self._start_new_step(observation=prompt)
        
        # Call the original Strands Agent async logic
        result = await super().invoke_async(prompt, **kwargs)
        
        # Extract relevant information from the result (same logic as __call__)
        model_response = ""
        action = None
        
        if hasattr(result, 'message') and result.message:
            if hasattr(result.message, 'content') and isinstance(result.message.content, list):
                for content_block in result.message.content:
                    if isinstance(content_block, dict) and 'text' in content_block:
                        model_response += content_block['text']
            elif hasattr(result.message, 'content'):
                model_response = str(result.message.content)
                
            action = result.message if hasattr(result, 'message') else result
        
        done = hasattr(result, 'stop_reason') and result.stop_reason in ['end_turn', 'stop_sequence']
        
        # Finish the current step
        self._finish_current_step(
            model_response=model_response,
            action=action, 
            done=done
        )
        
        return result
    
    def get_current_state(self) -> Step | None:
        """Get the current step state."""
        if self._trajectory.steps:
            return self._trajectory.steps[-1]
        return self._current_step
    
    def update_step_reward(self, reward: float):
        """Update the reward for the current or last step."""
        if self._current_step is not None:
            self._current_step.reward = reward
        elif self._trajectory.steps:
            self._trajectory.steps[-1].reward = reward
            # Update trajectory total reward
            self._trajectory.reward = sum(step.reward for step in self._trajectory.steps)
    
    def update_step_info(self, info: dict):
        """Update the info for the current or last step."""
        if self._current_step is not None:
            self._current_step.info.update(info)
        elif self._trajectory.steps:
            self._trajectory.steps[-1].info.update(info)