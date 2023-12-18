import logging
import threading
from typing import Any, Dict, List, Mapping, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    else:
        raise ValueError(f"Got unknown type {message}")
    return message_dict


class ErnieBotChat(BaseChatModel):
    """`ERNIE-Bot` large language model.

    ERNIE-Bot is a large language model developed by Baidu,
    covering a huge amount of Chinese data.

    To use, you should have the `ernie_client_id` and `ernie_client_secret` set,
    or set the environment variable `ERNIE_CLIENT_ID` and `ERNIE_CLIENT_SECRET`.

    Note:
    access_token will be automatically generated based on client_id and client_secret,
    and will be regenerated after expiration (30 days).

    Default model is `ERNIE-Bot-turbo`,
    currently supported models are `ERNIE-Bot-turbo`, `ERNIE-Bot`, `ERNIE-Bot-8K`,
    `ERNIE-Bot-4`, `ERNIE-Bot-turbo-AI`.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ErnieBotChat
            chat = ErnieBotChat(model_name='ERNIE-Bot')


    Deprecated Note:
    Please use `QianfanChatEndpoint` instead of this class.
    `QianfanChatEndpoint` is a more suitable choice for production.

    Always test your code after changing to `QianfanChatEndpoint`.

    Example of `QianfanChatEndpoint`:
        .. code-block:: python

            from langchain_community.chat_models import QianfanChatEndpoint
            qianfan_chat = QianfanChatEndpoint(model="ERNIE-Bot",
                endpoint="your_endpoint", qianfan_ak="your_ak", qianfan_sk="your_sk")

    """

    ernie_api_base: Optional[str] = None
    """Baidu application custom endpoints"""

    ernie_client_id: Optional[str] = None
    """Baidu application client id"""

    ernie_client_secret: Optional[str] = None
    """Baidu application client secret"""

    access_token: Optional[str] = None
    """access token is generated by client id and client secret,
    setting this value directly will cause an error"""

    model_name: str = "ERNIE-Bot-turbo"
    """model name of ernie, default is `ERNIE-Bot-turbo`.
      Currently supported `ERNIE-Bot-turbo`, `ERNIE-Bot`"""

    system: Optional[str] = None
    """system is mainly used for model character design,
    for example, you are an AI assistant produced by xxx company.
    The length of the system is limiting of 1024 characters."""

    request_timeout: Optional[int] = 60
    """request timeout for chat http requests"""

    streaming: Optional[bool] = False
    """streaming mode. not supported yet."""

    top_p: Optional[float] = 0.8
    temperature: Optional[float] = 0.95
    penalty_score: Optional[float] = 1

    _lock = threading.Lock()

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["ernie_api_base"] = get_from_dict_or_env(
            values, "ernie_api_base", "ERNIE_API_BASE", "https://aip.baidubce.com"
        )
        values["ernie_client_id"] = get_from_dict_or_env(
            values,
            "ernie_client_id",
            "ERNIE_CLIENT_ID",
        )
        values["ernie_client_secret"] = get_from_dict_or_env(
            values,
            "ernie_client_secret",
            "ERNIE_CLIENT_SECRET",
        )
        return values

    def _chat(self, payload: object) -> dict:
        base_url = f"{self.ernie_api_base}/rpc/2.0/ai_custom/v1/wenxinworkshop/chat"
        model_paths = {
            "ERNIE-Bot-turbo": "eb-instant",
            "ERNIE-Bot": "completions",
            "ERNIE-Bot-8K": "ernie_bot_8k",
            "ERNIE-Bot-4": "completions_pro",
            "ERNIE-Bot-turbo-AI": "ai_apaas",
            "BLOOMZ-7B": "bloomz_7b1",
            "Llama-2-7b-chat": "llama_2_7b",
            "Llama-2-13b-chat": "llama_2_13b",
            "Llama-2-70b-chat": "llama_2_70b",
        }
        if self.model_name in model_paths:
            url = f"{base_url}/{model_paths[self.model_name]}"
        else:
            raise ValueError(f"Got unknown model_name {self.model_name}")

        resp = requests.post(
            url,
            timeout=self.request_timeout,
            headers={
                "Content-Type": "application/json",
            },
            params={"access_token": self.access_token},
            json=payload,
        )
        return resp.json()

    def _refresh_access_token_with_lock(self) -> None:
        with self._lock:
            logger.debug("Refreshing access token")
            base_url: str = f"{self.ernie_api_base}/oauth/2.0/token"
            resp = requests.post(
                base_url,
                timeout=10,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                params={
                    "grant_type": "client_credentials",
                    "client_id": self.ernie_client_id,
                    "client_secret": self.ernie_client_secret,
                },
            )
            self.access_token = str(resp.json().get("access_token"))

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            raise ValueError("`streaming` option currently unsupported.")

        if not self.access_token:
            self._refresh_access_token_with_lock()
        payload = {
            "messages": [_convert_message_to_dict(m) for m in messages],
            "top_p": self.top_p,
            "temperature": self.temperature,
            "penalty_score": self.penalty_score,
            "system": self.system,
            **kwargs,
        }
        logger.debug(f"Payload for ernie api is {payload}")
        resp = self._chat(payload)
        if resp.get("error_code"):
            if resp.get("error_code") == 111:
                logger.debug("access_token expired, refresh it")
                self._refresh_access_token_with_lock()
                resp = self._chat(payload)
            else:
                raise ValueError(f"Error from ErnieChat api response: {resp}")
        return self._create_chat_result(resp)

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        if "function_call" in response:
            additional_kwargs = {
                "function_call": dict(response.get("function_call", {}))
            }
        else:
            additional_kwargs = {}
        generations = [
            ChatGeneration(
                message=AIMessage(
                    content=response.get("result"),
                    additional_kwargs={**additional_kwargs},
                )
            )
        ]
        token_usage = response.get("usage", {})
        llm_output = {"token_usage": token_usage, "model_name": self.model_name}
        return ChatResult(generations=generations, llm_output=llm_output)

    @property
    def _llm_type(self) -> str:
        return "ernie-bot-chat"
