from abc import ABC
from typing import Any, Dict, Optional, Tuple

from pydantic import Field
import json
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.memory.utils import get_prompt_input_key
from langchain.schema import BaseChatMessageHistory, BaseMemory


class BaseChatMemory(BaseMemory, ABC):
    chat_memory: BaseChatMessageHistory = Field(default_factory=ChatMessageHistory)
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    intermediate_steps_key: Optional[str] = None
    return_messages: bool = False
    intermediate_steps: Optional[str] = None

    def _get_input_output(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> Tuple[str, str]:
        
        if "intermediate_steps" in outputs.keys():
            intermediate_steps = outputs["intermediate_steps"]
            outputs.pop("intermediate_steps")
            self.intermediate_steps = intermediate_steps

        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            if len(outputs) != 1:
                raise ValueError(f"One output key expected, got {outputs.keys()}")
            output_key = list(outputs.keys())[0]
        else:
            output_key = self.output_key
        return inputs[prompt_input_key], outputs[output_key]

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.chat_memory.add_user_message(input_str)
        self.chat_memory.add_ai_message(output_str)
        if self.intermediate_steps:
            self.chat_memory.add_agent_message(json.dumps([x[0].to_dict() for x in self.intermediate_steps]))

    def clear(self) -> None:
        """Clear memory contents."""
        self.chat_memory.clear()

