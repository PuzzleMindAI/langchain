"""Hugging Face Chat Wrapper."""


from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Literal,
    Sequence,
    Type,
    Union,
    Tuple,
    cast
)


from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.pydantic_v1 import BaseModel,root_validator
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.language_models import LanguageModelInput
from langchain_core.utils.function_calling import convert_to_openai_tool

from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.llms.huggingface_text_gen_inference import (
    HuggingFaceTextGenInference,
)


DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful, and honest assistant."""

def _convert_message_to_chat_message(
        message: BaseMessage,
    ) -> Dict:
        if isinstance(message, ChatMessage):
            return dict(role=message.role, content=message.content)
        elif isinstance(message, HumanMessage):
            return dict(role="user", content=message.content)
        elif isinstance(message, AIMessage):
            if "tool_calls" in message.additional_kwargs:
                tool_calls = [
                    {
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        }
                    }
                    for tc in message.additional_kwargs["tool_calls"]
                ]
            else:
                tool_calls = None
            return {
                "role": "assistant",
                "content": message.content,
                "tool_calls": tool_calls,
            }
        elif isinstance(message, SystemMessage):
            return dict(role="system", content=message.content)
        elif isinstance(message, ToolMessage):
            return {
                "role": "tool",
                "content": message.content,
                "name": message.name,
            }
        else:
            raise ValueError(f"Got unknown type {message}")
def _convert_TGI_message_to_LC_message(
    _message: Dict,
) -> BaseMessage:
    role = _message.role
    assert role == "assistant", f"Expected role to be 'assistant', got {role}"
    content = cast(str, _message.content)
    if content is None:
        content=""
    additional_kwargs: Dict = {}
    if tool_calls := _message.tool_calls:
        if 'parameters' in tool_calls[0]['function']:
                # Convert list to string and replace all single quotes with double quotes
                var = str(tool_calls[0]['function'].pop('parameters')).replace("'", '"')
                tool_calls[0]['function']['arguments'] = var
        additional_kwargs["tool_calls"] =tool_calls
    return AIMessage(content=content, additional_kwargs=additional_kwargs)
class ChatHuggingFace(BaseChatModel):
    """
    Wrapper for using Hugging Face LLM's as ChatModels.

    Works with `HuggingFaceTextGenInference`, `HuggingFaceEndpoint`,
    and `HuggingFaceHub` LLMs.

    Upon instantiating this class, the model_id is resolved from the url
    provided to the LLM, and the appropriate tokenizer is loaded from
    the HuggingFace Hub.

    Adapted from: https://python.langchain.com/docs/integrations/chat/llama2_chat
    """

    llm: Any
    """LLM, must be of type HuggingFaceTextGenInference, HuggingFaceEndpoint, or 
        HuggingFaceHub."""
    system_message: SystemMessage = SystemMessage(content=DEFAULT_SYSTEM_PROMPT)
    tokenizer: Any = None
    model_id: Optional[str] = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        from transformers import AutoTokenizer

        self._resolve_model_id()

        self.tokenizer = (
            AutoTokenizer.from_pretrained(self.model_id)
            if self.tokenizer is None
            else self.tokenizer
        )

    @root_validator()
    def validate_llm(cls, values: dict) -> dict:
        if not isinstance(
            values["llm"],
            (HuggingFaceHub,HuggingFaceTextGenInference,HuggingFaceEndpoint),
        ):
            raise TypeError(
                "Expected llm to be one of HuggingFaceTextGenInference, "
                f"HuggingFaceEndpoint, HuggingFaceHub, received {type(values['llm'])}"
            )
        return values

    def _create_chat_result(self, response: Dict) -> ChatResult:
        generations = []
        
        finish_reason = response.choices[0].finish_reason
        gen = ChatGeneration(
            message=_convert_TGI_message_to_LC_message(response.choices[0].message),
            generation_info={"finish_reason": finish_reason},
        )
        generations.append(gen)
        token_usage = response.usage
        llm_output = {"token_usage": token_usage, "model": self.llm.inference_server_url}
        return ChatResult(generations=generations, llm_output=llm_output)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if  isinstance(self.llm,HuggingFaceTextGenInference):
           
            message_dicts = self._create_message_dicts(messages, stop)
           
            answer=self.llm.client.chat(messages=message_dicts,**kwargs)
            return(self._create_chat_result(answer))
        else:
            llm_input = self._to_chat_prompt(messages)
            llm_result = self.llm._generate(
                prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
            )
            return self._to_chat_result(llm_result)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm_input = self._to_chat_prompt(messages)
        llm_result = await self.llm._agenerate(
            prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
        )
        return self._to_chat_result(llm_result)

    def _to_chat_prompt(
        self,
        messages: List[BaseMessage],
    ) -> str:
        """Convert a list of messages into a prompt format expected by wrapped LLM."""
        if not messages:
            raise ValueError("At least one HumanMessage must be provided!")

        if not isinstance(messages[-1], HumanMessage):
            raise ValueError("Last message must be a HumanMessage!")

        messages_dicts = [self._to_chatml_format(m) for m in messages]

        return self.tokenizer.apply_chat_template(
            messages_dicts, tokenize=False, add_generation_prompt=True
        )

    def _to_chatml_format(self, message: BaseMessage) -> dict:
        """Convert LangChain message to ChatML format."""

        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

        return {"role": role, "content": message.content}
    
    @staticmethod
    def _to_chat_result(llm_result: LLMResult) -> ChatResult:
        chat_generations = []

        for g in llm_result.generations[0]:
            chat_generation = ChatGeneration(
                message=AIMessage(content=g.text), generation_info=g.generation_info
            )
            chat_generations.append(chat_generation)

        return ChatResult(
            generations=chat_generations, llm_output=llm_result.llm_output
        )

    def _resolve_model_id(self) -> None:
        """Resolve the model_id from the LLM's inference_server_url"""

        from huggingface_hub import list_inference_endpoints

        available_endpoints = list_inference_endpoints("*")
        if isinstance(self.llm, HuggingFaceHub) or (
            hasattr(self.llm, "repo_id") and self.llm.repo_id
        ):
            self.model_id = self.llm.repo_id
            return
        elif isinstance(self.llm, HuggingFaceTextGenInference):
            endpoint_url: Optional[str] = self.llm.inference_server_url
        else:
            endpoint_url = self.llm.endpoint_url

        for endpoint in available_endpoints:
            if endpoint.url == endpoint_url:
                self.model_id = endpoint.repository

        if not self.model_id:
            raise ValueError(
                "Failed to resolve model_id:"
                f"Could not find model id for inference server: {endpoint_url}"
                "Make sure that your Hugging Face token has access to the endpoint."
            )
    def _convert_delta_to_message_chunk(
        _delta: Dict, default_class: Type[BaseMessageChunk]
    ) -> BaseMessageChunk:
        role = _delta.get("role")
        content = _delta.get("content") or ""
        if role == "user" or default_class == HumanMessageChunk:
            return HumanMessageChunk(content=content)
        elif role == "assistant" or default_class == AIMessageChunk:
            additional_kwargs: Dict = {}
            if tool_calls := _delta.get("tool_calls"):
                additional_kwargs["tool_calls"] = tool_calls
            return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
        elif role == "system" or default_class == SystemMessageChunk:
            return SystemMessageChunk(content=content)
        elif role or default_class == ChatMessageChunk:
            return ChatMessageChunk(content=content, role=role)
        else:
            return default_class(content=content)
  
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[dict, str, Literal["auto", "none"], bool]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            tool_choice: Which tool to require the model to call.
                Must be the name of the single provided function or
                "auto" to automatically determine which function to call
                (if any), or a dict of the form:
                {"type": "function", "function": {"name": <<tool_name>>}}.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None and tool_choice:
            if len(formatted_tools) != 1:
                raise ValueError(
                    "When specifying `tool_choice`, you must provide exactly one "
                    f"tool. Received {len(formatted_tools)} tools."
                )
            if isinstance(tool_choice, str):
                if tool_choice not in ("auto", "none"):
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
            elif isinstance(tool_choice, bool):
                tool_choice = formatted_tools[0]
            elif isinstance(tool_choice, dict):
                if (
                    formatted_tools[0]["function"]["name"]
                    != tool_choice["function"]["name"]
                ):
                    raise ValueError(
                        f"Tool choice {tool_choice} was specified, but the only "
                        f"provided tool was {formatted_tools[0]['function']['name']}."
                    )
            else:
                raise ValueError(
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)
    
    

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        
        message_dicts = [_convert_message_to_chat_message(m) for m in messages]
        return message_dicts

    @property
    def _llm_type(self) -> str:
        return "huggingface-chat-wrapper"
