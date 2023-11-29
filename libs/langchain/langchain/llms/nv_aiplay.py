## NOTE: This class is intentionally implemented to subclass either ChatModel or LLM for
##  demonstrative purposes and to make it function as a simple standalone file.

from __future__ import annotations

from langchain.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    CallbackManager,
)
from langchain.pydantic_v1 import BaseModel, Field, root_validator
from langchain.schema.messages import BaseMessage, ChatMessageChunk
from langchain.schema.output import ChatGenerationChunk, GenerationChunk
from langchain.utils import get_from_dict_or_env

try:
    ## if running as part of package
    from .base import LLM

    STANDALONE = False
except Exception:
    ## if running as standalone file
    from langchain.chat_models.base import SimpleChatModel
    from langchain.llms.base import LLM

    STANDALONE = True

import asyncio
import json
import logging
import re
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import aiohttp
import requests
from requests.models import Response

logger = logging.getLogger(__name__)


class ClientModel(BaseModel):
    """
    Custom BaseModel subclass with some desirable properties for subclassing
    """

    saved_parent: Optional[ClientModel] = None

    def __init__(self, *args: Sequence, **kwargs: Any[str, Any]):
        super().__init__(*args, **kwargs)

    def subscope(self, *args: Sequence, **kwargs: Any) -> Any:
        """Create a new ClientModel with the same values but new arguments"""
        named_args = dict({k: v for k, v in zip(getattr(self, "arg_keys", []), args)})
        named_args = {**named_args, **kwargs}
        out = self.copy(update=named_args)
        for k, v in self.__dict__.items():
            if isinstance(v, ClientModel):
                setattr(out, k, v.subscope(*args, **kwargs))
        out.saved_parent = self
        return out

    def get(self, key: str) -> Any:
        """Get a value from the ClientModel, using it like a dictionary"""
        return getattr(self, key)

    def transfer_state(self, other: Optional[ClientModel]) -> None:
        """Transfer state from one ClientModel to another"""
        if other is None:
            return
        for k, v in self.__dict__.items():
            if k in getattr(self, "state_vars", []):
                setattr(other, k, v)
            elif hasattr(v, "transfer_state"):
                other_sub = getattr(other, k, None)
                if other_sub is not None:
                    v.transfer_state(other_sub)

    def __enter__(self) -> ClientModel:
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        self.transfer_state(self.saved_parent)
        self.saved_parent = None


class NVCRModel(ClientModel):

    """
    Underlying Client for interacting with the AI Playground API.
    Leveraged by the NVAIPlayBaseModel to provide a simple requests-oriented interface.
    Direct abstraction over NGC-recommended streaming/non-streaming Python solutions.

    NOTE: AI Playground does not currently support raw text continuation.
    TODO: Add support for raw text continuation for arbitrary (non-AIP) nvcf functions.
    """

    ## Core defaults. These probably should not be changed
    fetch_url_format: str = Field("https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/")
    call_invoke_base: str = Field("https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions")
    get_session_fn: Callable = Field(requests.Session)
    get_asession_fn: Callable = Field(aiohttp.ClientSession)

    ## Populated on construction/validation
    nvapi_key: Optional[str]
    is_staging: Optional[bool]
    available_models: Optional[Dict[str, str]]

    ## Generation arguments
    max_tries: int = Field(5, ge=1)
    stop: Union[str, List[str]] = Field([])
    headers = dict(
        call={"Authorization": "Bearer {nvapi_key}", "Accept": "application/json"},
        stream={
            "Authorization": "Bearer {nvapi_key}",
            "Accept": "text/event-stream",
            "content-type": "application/json",
        },
    )

    ## Status Tracking Variables. Updated Progressively
    last_inputs: Optional[dict] = None
    last_response: Optional[Any] = None
    last_msg: dict = {}
    state_vars: Sequence[str] = ["last_inputs", "last_response", "last_msg"]

    @root_validator()
    def validate_model(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and update model arguments, including API key and formatting"""
        values["nvapi_key"] = get_from_dict_or_env(values, "nvapi_key", "NVAPI_KEY")
        if "nvapi-" not in values.get("nvapi_key", ""):
            raise ValueError("Invalid NVAPI key detected. Should start with `nvapi-`")
        values["is_staging"] = "nvapi-stg-" in values["nvapi_key"]
        for header in values["headers"].values():
            if "{nvapi_key}" in header["Authorization"]:
                header["Authorization"] = header["Authorization"].format(
                    nvapi_key=values["nvapi_key"]
                )
        if isinstance(values["stop"], str):
            values["stop"] = [values["stop"]]
        return values

    def __init__(self, *args: Sequence, **kwargs: Any):
        """Useful to define custom operations on construction after validation"""
        super().__init__(*args, **kwargs)
        self.fetch_url_format = self._stagify(self.fetch_url_format)
        self.call_invoke_base = self._stagify(self.call_invoke_base)
        try:
            self.available_models = self.get_available_models()
        except Exception as e:
            raise Exception("Error retrieving model list. Verify your NVAPI key") from e

    def _stagify(self, path: str) -> str:
        """Helper method to switch between staging and production endpoints"""
        if self.is_staging and "stg.api" not in path:
            return path.replace("api", "stg.api")
        if not self.is_staging and "stg.api" in path:
            return path.replace("stg.api", "api")
        return path

    ####################################################################################
    ## Core utilities for posting and getting from NVCR

    def _post(self, invoke_url: str, payload: dict = {}) -> Tuple[Response, Any]:
        """Method for posting to the AI Playground API."""
        self.last_inputs = dict(
            url=invoke_url,
            headers=self.headers["call"].copy(),
            json=payload,
            stream=False,
        )
        session = self.get_session_fn()
        self.last_response = session.post(**self.last_inputs)
        return self.last_response, session

    def _get(self, invoke_url: str, payload: dict = {}) -> Tuple[Response, Any]:
        """Method for getting from the AI Playground API."""
        self.last_inputs = dict(
            url=invoke_url,
            headers=self.headers["call"].copy(),
            json=payload,
            stream=False,
        )
        session = self.get_session_fn()
        self.last_response = session.get(**self.last_inputs)
        return self.last_response, session

    def _wait(self, response: Response, session: Any) -> Response:
        """Wait for a response from API after an initial response is made."""
        i = 1
        while response.status_code == 202:
            request_id = response.headers.get("NVCF-REQID", "")
            response = session.get(
                self.fetch_url_format + request_id,
                headers=self.headers["call"].copy(),
            )
            if response.status_code == 202:
                try:
                    body = response.json()
                except ValueError:
                    body = str(response)
                if i > self.max_tries:
                    raise ValueError(f"Failed to get response with {i} tries: {body}")
            response.raise_for_status()
            i += 1
        return response

    ####################################################################################
    ## Simple query interface to show the set of model options

    def query(self, invoke_url: str, payload: dict = {}) -> dict:
        """Simple method for an end-to-end get query. Returns result dictionary"""
        response, session = self._get(invoke_url, payload)
        response = self._wait(response, session)
        output = self._process_response(response)[0]
        return output

    def _process_response(self, response: Union[str, Response]) -> List[dict]:
        """General-purpose response processing for single responses and streams"""
        if hasattr(response, "json"):  ## For single response (i.e. non-streaming)
            try:
                return [response.json()]
            except json.JSONDecodeError:
                response = str(response.__dict__)
        if isinstance(response, str):  ## For set of responses (i.e. streaming)
            msg_list = []
            for msg in response.split("\n\n"):
                if "{" not in msg:
                    continue
                msg_list += [json.loads(msg[msg.find("{") :])]
            return msg_list
        raise ValueError(f"Received ill-formed response: {response}")

    def get_available_models(self) -> dict:
        """Get a dictionary of available models from the AI Playground API."""
        invoke_url = self._stagify("https://api.nvcf.nvidia.com/v2/nvcf/functions")
        return {v["name"]: v["id"] for v in self.query(invoke_url)["functions"]}

    def _get_invoke_url(
        self, model_name: Optional[str] = None, invoke_url: Optional[str] = None
    ) -> str:
        """Helper method to get invoke URL from a model name, URL, or endpoint stub"""
        if not invoke_url:
            if not model_name:
                raise ValueError("URL or model name must be specified to invoke")
            available_models = self.available_models or self.get_available_models()
            if model_name in available_models:
                invoke_url = available_models.get(model_name)
            else:
                for k, v in available_models.items():
                    if model_name in k:
                        invoke_url = v
                        break
        if not invoke_url:
            raise ValueError(f"Unknown model name {model_name} specified")
        if "http" not in invoke_url:
            invoke_url = f"{self.call_invoke_base}/{invoke_url}"
        return invoke_url

    ####################################################################################
    ## Generation interface to allow users to generate new values from endpoints

    def get_req_generation(
        self,
        model_name: Optional[str] = None,
        payload: dict = {},
        invoke_url: Optional[str] = None,
    ) -> dict:
        """Method for an end-to-end post query with NVCR post-processing."""
        invoke_url = self._get_invoke_url(model_name, invoke_url)
        if payload.get("stream", False) is True:
            payload = {**payload, "stream": False}
        response, session = self._post(invoke_url, payload)
        response = self._wait(response, session)
        output, _ = self.postprocess(response)
        return output

    def postprocess(self, response: Union[str, Response]) -> Tuple[dict, bool]:
        """Parses a response from the AI Playground API.
        Strongly assumes that the API will return a single response.
        """
        msg_list = self._process_response(response)
        msg, is_stopped = self._aggregate_msgs(msg_list)
        msg, is_stopped = self._early_stop_msg(msg, is_stopped)
        return msg, is_stopped

    def _aggregate_msgs(self, msg_list: Sequence[dict]) -> Tuple[dict, bool]:
        """Dig into retrieved message and tease out ['choices'][0]['delta'/'message']"""
        content_buffer = ""
        content_holder = {"content": ""}
        is_stopped = False
        for msg in msg_list:
            self.last_msg = msg
            msg = msg.get("choices", [{}])[0]
            is_stopped = msg.get("finish_reason", "") == "stop"
            msg = msg.get("delta", msg.get("message", {"content": ""}))
            content_holder = msg
            content_buffer += msg.get("content", "")
            if is_stopped:
                break
        content_holder["content"] = content_buffer
        return content_holder, is_stopped

    def _early_stop_msg(self, msg: dict, is_stopped: bool) -> Tuple[dict, bool]:
        """Try to early-terminate streaming or generation by iterating over stop list"""
        content = msg.get("content", "")
        if content and self.stop:
            for stop_str in self.stop:
                if stop_str and stop_str in content:
                    msg["content"] = content[: content.find(stop_str) + 1]
                    is_stopped = True
        return msg, is_stopped

    ####################################################################################
    ## Streaming interface to allow you to iterate through progressive generations

    def get_req_stream(
        self,
        model: Optional[str] = None,
        payload: dict = {},
        invoke_url: Optional[str] = None,
    ) -> Iterator:
        invoke_url = self._get_invoke_url(model, invoke_url)
        if payload.get("stream", True) is False:
            payload = {**payload, "stream": True}
        self.last_inputs = dict(
            url=invoke_url,
            headers=self.headers["stream"].copy(),
            json=payload,
            stream=True,
        )
        self.last_response = self.get_session_fn().post(**self.last_inputs)
        for line in self.last_response.iter_lines():
            if line and line.strip() != b"data: [DONE]":
                line = line.decode("utf-8")
                msg, final_line = self.postprocess(line)
                yield msg
                if final_line:
                    break
            self.last_response.raise_for_status()

    ####################################################################################
    ## Asynchronous streaming interface to allow multiple generations to happen at once.

    async def get_req_astream(
        self,
        model: Optional[str] = None,
        payload: dict = {},
        invoke_url: Optional[str] = None,
    ) -> AsyncIterator:
        invoke_url = self._get_invoke_url(model, invoke_url)
        if payload.get("stream", True) is False:
            payload = {**payload, "stream": True}
        self.last_inputs = dict(
            url=invoke_url,
            headers=self.headers["stream"].copy(),
            json=payload,
        )
        async with self.get_asession_fn() as session:
            async with session.post(**self.last_inputs) as self.last_response:
                async for line in self.last_response.content.iter_any():
                    if line and line.strip() != b"data: [DONE]":
                        line = line.decode("utf-8")
                        msg, final_line = self.postprocess(line)
                        yield msg
                        if final_line:
                            break
                self.last_response.raise_for_status()


class NVAIPlayClient(ClientModel):
    """
    Higher-Level Client for interacting with AI Playground API with argument defaults.
    Is subclassed by NVAIPlayLLM/NVAIPlayChat to provide a simple LangChain interface.
    """

    client: NVCRModel = Field(NVCRModel)

    model_name: str = Field("llama2_13b")
    model: Optional[str] = Field(None)
    labels: dict = Field({})

    temperature: float = Field(0.2, le=1.0, gt=0.0)
    top_p: float = Field(0.7, le=1.0, ge=0.0)
    max_tokens: int = Field(1024, le=1024, ge=32)
    streaming: bool = Field(False)

    inputs: Any = Field([])
    stop: Union[Sequence[str], str] = Field([])

    gen_keys: Sequence[str] = Field(["temperature", "top_p", "max_tokens", "streaming"])
    arg_keys: Sequence[str] = Field(["inputs", "labels", "stop"])
    valid_roles: Sequence[str] = Field(["user", "system", "assistant", "context"])

    def __init__(self, *args: Sequence, **kwargs: Any):
        super().__init__(*args, **kwargs)

    @root_validator()
    def validate_model(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        values["client"] = values["client"](**values)
        model_name = values.get("model")
        model_name = model_name if model_name else values["model_name"]
        values["model_name"] = model_name
        values["model"] = model_name
        return values

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def available_models(self) -> List[str]:
        """List the available models that can be invoked"""
        return list(getattr(self.client, "available_models", {}).keys())

    # ## Default Call Behavior. Great for standalone use, but not for LangChain
    # def __call__(self, *args: Sequence, **kwargs: Any):
    #     '''
    #     Calls the AI Playground API with the given arguments.
    #     Directs to `generate` or `stream` depending on the `stream` argument.
    #     '''
    #     stream = kwargs.get('stream', kwargs.get('streaming', self.streaming))
    #     out_fn = self.get_stream if stream else self.get_generation
    #     return out_fn(*args, **kwargs)

    def get_generation(self, *args: Sequence, **kwargs: Any) -> dict:
        """Call to client generate method with call scope"""
        with self.subscope(*args, **kwargs) as call:
            payload = call.get_payload(stream=False)
            out = call.client.get_req_generation(call.model_name, payload=payload)
        return out

    def get_stream(self, *args: Sequence, **kwargs: Any) -> Iterator:
        """Call to client stream method with call scope"""
        with self.subscope(*args, **kwargs) as call:
            payload = call.get_payload(stream=True)
            out = call.client.get_req_stream(call.model_name, payload=payload)
        return out

    def get_astream(self, *args: Sequence, **kwargs: Any) -> AsyncIterator:
        """Call to client astream method with call scope"""
        with self.subscope(*args, **kwargs) as call:
            payload = call.get_payload(stream=True)
            out = call.client.get_req_astream(call.model_name, payload=payload)
        return out

    def get_payload(self, *args: Sequence, **kwargs: Any) -> dict:
        """Generates payload for the NVAIPlayClient API to send to service."""

        def k_map(k: str) -> str:
            return k if k != "streaming" else "stream"

        out = {**self.preprocess(), **{k_map(k): self.get(k) for k in self.gen_keys}}
        return out

    def preprocess(self) -> dict:
        """Prepares a message or list of messages for the payload"""
        if (
            isinstance(self.inputs, str)
            or not hasattr(self.inputs, "__iter__")
            or isinstance(self.inputs, BaseMessage)
        ):
            self.inputs = [self.inputs]
        messages = [self.prep_msg(m) for m in self.inputs]
        labels = self.labels
        if labels:
            messages += [{"labels": labels, "role": "assistant"}]
        return {"messages": messages}

    def prep_msg(self, msg: Union[str, dict, BaseMessage]) -> dict:
        """Helper Method: Ensures a message is a dictionary with a role and content."""
        if isinstance(msg, str):
            return dict(role="user", content=msg)
        if isinstance(msg, dict):
            if msg.get("role", "") not in self.valid_roles:
                raise ValueError(f"Unknown message role \"{msg.get('role', '')}\"")
            if msg.get("content", None) is None:
                raise ValueError(f"Message {msg} has no content")
            return msg
        raise ValueError(f"Unknown message received: {msg} of type {type(msg)}")


class NVAIPlayBaseModel(NVAIPlayClient):
    """
    Base class for NVIDIA AI Playground models which can interface with NVAIPlayClient.
    To be subclassed by NVAIPlayLLM/NVAIPlayChat by combining with LLM/SimpleChatModel.
    """

    @property
    def _llm_type(self) -> str:
        """Return type of NVIDIA AI Playground Interface."""
        return "nvidia_ai_playground"

    def _call(
        self,
        messages: Union[List[BaseMessage], str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> str:
        """hook for LLM/SimpleChatModel. Allows for easy standard/streaming calls"""
        inputs = self.custom_preprocess(messages)
        labels = kwargs.get("labels", self.labels)
        if kwargs.get("streaming", self.streaming) or stop:
            buffer = ""
            for chunk in self._stream(
                messages=messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                buffer += chunk if isinstance(chunk, str) else chunk.text
            responses = {"content": buffer}
        else:
            responses = self.get_generation(inputs, labels=labels, stop=stop, **kwargs)
        outputs = self.custom_postprocess(responses)
        return outputs

    def _get_filled_chunk(
        self, text: str, role: Optional[str] = "assistant"
    ) -> Union[GenerationChunk, ChatGenerationChunk]:
        """LLM and BasicChatModel have different streaming chunk specifications"""
        if isinstance(self, LLM):
            return GenerationChunk(text=text)
        return ChatGenerationChunk(message=ChatMessageChunk(content=text, role=role))

    def _stream(
        self,
        messages: Union[List[BaseMessage], str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> Iterator[Union[GenerationChunk, ChatGenerationChunk]]:
        """Allows streaming to model!"""
        inputs = self.custom_preprocess(messages)
        labels = kwargs.get("labels", self.labels)
        if not stop:
            stop = getattr(self.client, "stop")
        for response in self.get_stream(inputs, labels=labels, stop=stop, **kwargs):
            chunk = self._get_filled_chunk(self.custom_postprocess(response))
            yield chunk
            if run_manager:
                if isinstance(
                    run_manager, (AsyncCallbackManager, AsyncCallbackManagerForLLMRun)
                ):
                    ## Edge case from LLM/SimpleChatModel default async methods
                    asyncio.run(run_manager.on_llm_new_token(chunk.text, chunk=chunk))
                else:
                    run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    async def _astream(
        self,
        messages: Union[List[BaseMessage], str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManager] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Union[GenerationChunk, ChatGenerationChunk]]:
        inputs = self.custom_preprocess(messages)
        labels = kwargs.get("labels", self.labels)
        if not stop:
            stop = getattr(self.client, "stop")
        async for response in self.get_astream(
            inputs, labels=labels, stop=stop, **kwargs
        ):
            chunk = self._get_filled_chunk(self.custom_postprocess(response))
            yield chunk
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    def custom_preprocess(self, msgs: Union[str, Sequence]) -> List[Dict[str, str]]:
        if isinstance(msgs, str):
            msgs = re.split("///ROLE ", msgs.strip())
            if msgs[0] == "":
                msgs = msgs[1:]
        elif not hasattr(msgs, "__iter__") or isinstance(msgs, BaseMessage):
            msgs = [msgs]
        out = [self.preprocess_msg(m) for m in msgs]
        return out

    def preprocess_msg(
        self, msg: Union[str, Sequence[str], dict, BaseMessage]
    ) -> Dict[str, str]:
        ## Support for just simple string inputs of ///ROLE SYS etc. inputs
        if isinstance(msg, str):
            msg_split = re.split("SYS: |USER: |AGENT: |CONTEXT:", msg)
            if len(msg_split) == 1:
                return {"role": "user", "content": msg}
            else:
                role_convert = {"AGENT": "assistant", "SYS": "system"}
                role, content = msg.split(": ")[:2]
                role = role_convert.get(role, "user")
                return {"role": role, "content": content}
        ## Support for tuple inputs
        if type(msg) in (list, tuple):
            return {"role": msg[0], "content": msg[1]}
        ## Support for manually-specified default inputs to AI Playground
        if isinstance(msg, dict) and msg.get("content"):
            msg["role"] = msg.get("role", "user")
            return msg
        ## Support for LangChain Messages
        if hasattr(msg, "content"):
            role_convert = {"ai": "assistant", "system": "system"}
            role = getattr(msg, "type")
            cont = getattr(msg, "content")
            role = role_convert.get(role, "user")
            if hasattr(msg, "role"):
                cont = f"{getattr(msg, 'role')}: {cont}"
            return {"role": role, "content": cont}
        raise ValueError(f"Invalid message: {repr(msg)} of type {type(msg)}")

    def custom_postprocess(self, msg: dict) -> str:
        if "content" in msg:
            return msg["content"]
        logger.warning(
            f"Got ambiguous message in postprocessing; returning as-is: msg = {msg}"
        )
        return str(msg)


################################################################################


class NVAIPlayLLM(NVAIPlayBaseModel, LLM):
    pass


if STANDALONE:

    class NVAIPlayChat(NVAIPlayBaseModel, SimpleChatModel):
        pass


################################################################################


class LlamaLLM(NVAIPlayLLM):
    model_name: str = Field("llama2_13b")


class MistralLLM(NVAIPlayLLM):
    model_name: str = Field("mistral")


class SteerLM(NVAIPlayLLM):
    model_name: str = Field("gpt_steerlm_8b")
    labels: dict = Field(
        {
            "creativity": 5,
            "helpfulness": 5,
            "humor": 5,
            "quality": 5,
        }
    )


class NemotronQA(NVAIPlayLLM):
    model_name: str = Field("gpt_qa_8b")


if STANDALONE:

    class LlamaChat(NVAIPlayChat):
        model_name: str = Field("llama2_13b")

    class MistralChat(NVAIPlayChat):
        model_name: str = Field("mistral")

    class SteerLMChat(NVAIPlayChat):
        model_name: str = Field("gpt_steerlm_8b")
        labels: dict = Field(
            {
                "creativity": 5,
                "helpfulness": 5,
                "humor": 5,
                "quality": 5,
            }
        )

    class NemotronQAChat(NVAIPlayChat):
        model_name: str = Field("gpt_qa_8b")
