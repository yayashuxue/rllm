import asyncio

import numpy as np
import openai
from transformers import AutoTokenizer

from rllm.parser.chat_template.parser import ChatTemplateParser
from rllm.router.router import Router


class RolloutEngine:
    def __init__(self, engine_name: str = "openai", tokenizer: AutoTokenizer = None, chat_parser: ChatTemplateParser = None, **kwargs):
        self.engine_name = engine_name

        self.tokenizer = tokenizer
        self.chat_parser = chat_parser
        if self.chat_parser is None and self.tokenizer is not None:
            self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer, disable_thinking=kwargs.get("disable_thinking", False))
        elif self.chat_parser is not None and self.tokenizer is None:
            self.tokenizer = self.chat_parser.tokenizer

        self.api_retries = kwargs.get("api_retries", 3)

        if self.engine_name == "openai":
            self.client = openai.AsyncOpenAI(**kwargs.get("openai_kwargs", {}))

        elif self.engine_name == "verl":
            assert self.chat_parser is not None and self.tokenizer is not None, "ChatTemplateParser and tokenizer are required for verl engine"  # is this passed from the trainer?
            self.rollout_manager = kwargs.get("rollout_manager", None)
            self.config = kwargs.get("config", None)
            self.max_prompt_length = self.config.data.max_prompt_length
            self.router = Router(config=self.config, tokenizer=self.tokenizer, addresses=self.rollout_manager.server_addresses)
        else:
            raise NotImplementedError(f"Engine type '{self.engine_name}' not supported")

    def wake_up(self):
        """Wake up the rollout engine (for verl engine)"""
        if hasattr(self, "rollout_manager") and hasattr(self.rollout_manager, "wake_up"):
            self.rollout_manager.wake_up()

    def sleep(self):
        """Put the rollout engine to sleep (for verl engine)"""
        if hasattr(self, "rollout_manager") and hasattr(self.rollout_manager, "sleep"):
            self.rollout_manager.sleep()

    async def get_model_response(self, messages: list[dict], **kwargs):
        """
        Compute model response asynchronously based on the engine type.

        This function is multithread safe and routes the request to the appropriate
        engine-specific handler.

        Args:
            messages: The input messages to send to the model
            **kwargs: Additional arguments to pass to the model

        Returns:
            The model's response text

        Raises:
            NotImplementedError: If the engine type is not supported
        """
        application_id = kwargs.pop("application_id", None)

        if self.engine_name == "openai":
            if self.chat_parser is None:
                return await self._get_response_openai_chat(messages, **kwargs)
            else:
                return await self._get_response_openai(messages, **kwargs)
        elif self.engine_name == "verl":
            assert application_id is not None, "application_id is required for verl engine"
            return await self._get_response_verl(messages, application_id, **kwargs)
        else:
            raise NotImplementedError(f"Engine type '{self.engine_name}' not supported")

    async def _get_response_verl(self, messages: list[dict], application_id: str, **kwargs) -> str:
        batch = self._convert_messages_verl([messages], **kwargs)

        if "max_tokens" in kwargs:
            batch.meta_info["max_tokens"] = kwargs["max_tokens"]

        output = await self.router.generate_sequences(batch, application_id=application_id, **kwargs)

        attn = output.batch["attention_mask"][0, self.max_prompt_length :]
        tokens = output.batch["responses"][0]

        # Find last index where attention == 1
        non_pad_indices = (attn == 1).nonzero(as_tuple=True)[0]
        if len(non_pad_indices) == 0:
            trimmed = tokens[:0]  # empty
        else:
            last_valid_idx = non_pad_indices[-1].item()
            trimmed = tokens[: last_valid_idx + 1]  # include the last valid token

        response = self.tokenizer.decode(trimmed, skip_special_tokens=False)

        pad_token = self.tokenizer.pad_token
        eos_token = self.tokenizer.eos_token
        response = response.replace(pad_token, "").replace(eos_token, "")
        return response

    async def _get_response_openai(self, messages: list[dict], **kwargs) -> str:
        """
        Get action from OpenAI Completions API asynchronously with retry logic.
        Must have a self.chat_parser.

        Args:
            messages: The input meassages for completions API
            **kwargs: Additional arguments to pass to the OpenAI API

        Returns:
            The response from OpenAI API
        """

        async def get_response(prompt_text: str):
            retries = self.api_retries
            while retries > 0:
                try:
                    response = await self.client.completions.create(
                        prompt=prompt_text,
                        timeout=3600,
                        **kwargs,
                    )
                    return response
                except openai.RateLimitError:
                    retries -= 1
                    if retries == 0:
                        return "Error: Rate limit reached and retries exhausted."
                    print("Sleep for 5 seconds for API limit.")
                    await asyncio.sleep(5)
                except Exception as e:
                    print("Error: ", e)
                    return f"Error processing content: {e}"

        prompt = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True)
        response = await get_response(prompt)
        if isinstance(response, openai.types.Completion):
            response = response.choices[0].text
        return response

    async def _get_response_openai_chat(self, messages: list[dict], **kwargs) -> str:
        """
        Get action from OpenAI Chat API asynchronously with retry logic.

        Args:
            messages: The input messages in text format for chat API
            **kwargs: Additional arguments to pass to the OpenAI API

        Returns:
            The response from OpenAI API
        """

        async def get_response(messages: list[dict]):
            retries = self.api_retries
            while retries > 0:
                try:
                    response = await self.client.chat.completions.create(
                        messages=messages,
                        timeout=3600,
                        **kwargs,
                    )
                    return response
                except openai.RateLimitError:
                    retries -= 1
                    if retries == 0:
                        return "Error: Rate limit reached and retries exhausted."
                    print("Sleep for 5 seconds for API limit.")
                    await asyncio.sleep(5)
                except Exception as e:
                    print("Error: ", e)
                    return f"Error processing content: {e}"

        response = await get_response(messages)
        if isinstance(response, openai.types.ChatCompletion):
            response = response.choices[0].message.content
        return response

    def _convert_messages_verl(self, messages, **kwargs):
        """
        Given a list of messages to convert to DataProto format in veRL

        Args:
            messagses: List of chat completion messages to convert
            **kwargs: Additional arguments

        Returns:
            DataProto object containing the converted prompts
        """
        from verl import DataProto
        from verl.protocol import union_two_dict
        from verl.utils.model import compute_position_id_with_mask
        from verl.utils.torch_functional import pad_sequence_to_length

        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        formatted_prompts = [self.chat_parser.parse(prompt, add_generation_prompt=True, is_first_msg=True) for prompt in messages]

        # Tokenize the final processed strings
        inputs = self.tokenizer(
            formatted_prompts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        self.tokenizer.padding_side = old_padding_side

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # pad to max sizes
        input_ids = pad_sequence_to_length(input_ids, max_seq_len=self.max_prompt_length, pad_token_id=self.tokenizer.pad_token_id, left_pad=True)
        attention_mask = pad_sequence_to_length(attention_mask, max_seq_len=self.max_prompt_length, pad_token_id=0, left_pad=True)
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        data = DataProto.from_dict(batch_dict)
        data.non_tensor_batch["formatted_prompts"] = np.array(formatted_prompts)

        # original_batch contains the extra info needed for generation
        if "meta_info" in kwargs and kwargs["meta_info"]:
            meta_info = kwargs["meta_info"]
            # only use the original_batch's meta_info since tensor_batch is from batch_dict and non_tensor_batch is not neeeded
            data.meta_info = union_two_dict(data.meta_info, meta_info)

        return data
