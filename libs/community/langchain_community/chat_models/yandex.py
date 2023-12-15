"""Wrapper around YandexGPT chat models."""
import logging
from typing import Any, Dict, List, Optional, Tuple, cast

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain_community.llms.utils import enforce_stop_tokens
from langchain_community.llms.yandex import _BaseYandexGPT

logger = logging.getLogger(__name__)


def _parse_message(role: str, text: str) -> Dict:
    return {"role": role, "text": text}


def _parse_chat_history(history: List[BaseMessage]) -> Tuple[List[Dict[str, str]], str]:
    """Parse a sequence of messages into history.

    Returns:
        A tuple of a list of parsed messages and an instruction message for the model.
    """
    chat_history = []
    for message in history:
        content = cast(str, message.content)
        if isinstance(message, HumanMessage):
            chat_history.append(_parse_message("user", content))
        if isinstance(message, AIMessage):
            chat_history.append(_parse_message("assistant", content))
        if isinstance(message, SystemMessage):
            chat_history.append(_parse_message("system", content))
    return chat_history


class ChatYandexGPT(_BaseYandexGPT, BaseChatModel):
    """Wrapper around YandexGPT large language models.

    There are two authentication options for the service account
    with the ``ai.languageModels.user`` role:
        - You can specify the token in a constructor parameter `iam_token`
        or in an environment variable `YC_IAM_TOKEN`.
        - You can specify the key in a constructor parameter `api_key`
        or in an environment variable `YC_API_KEY`.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatYandexGPT
            chat_model = ChatYandexGPT(iam_token="t1.9eu...")

    """

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate next turn in the conversation.
        Args:
            messages: The history of the conversation as a list of messages.
            stop: The list of stop words (optional).
            run_manager: The CallbackManager for LLM run, it's not used at the moment.

        Returns:
            The ChatResult that contains outputs generated by the model.

        Raises:
            ValueError: if the last message in the list is not from human.
        """
        try:
            import grpc
            from google.protobuf.wrappers_pb2 import DoubleValue, Int64Value
            from yandex.cloud.ai.foundation_models.v1.foundation_models_pb2 import (
                CompletionOptions,
                Message,
            )
            from yandex.cloud.ai.foundation_models.v1.foundation_models_service_pb2 import (  # noqa: E501
                CompletionRequest,
            )
            from yandex.cloud.ai.foundation_models.v1.foundation_models_service_pb2_grpc import (  # noqa: E501
                TextGenerationServiceStub,
            )
        except ImportError as e:
            raise ImportError(
                "Please install YandexCloud SDK" " with `pip install yandexcloud`."
            ) from e
        if not messages:
            raise ValueError(
                "You should provide at least one message to start the chat!"
            )
        message_history = _parse_chat_history(messages)
        channel_credentials = grpc.ssl_channel_credentials()
        channel = grpc.secure_channel(self.url, channel_credentials)
        request = CompletionRequest(
            model_uri=self.model_uri,
            completion_options=CompletionOptions(
                temperature=DoubleValue(value=self.temperature),
                max_tokens=Int64Value(value=self.max_tokens),
            ),
            messages=[Message(**message) for message in message_history],
        )
        stub = TextGenerationServiceStub(channel)
        res = stub.Completion(request, metadata=self._grpc_metadata)
        text = list(res)[0].alternatives[0].message.text
        text = text if stop is None else enforce_stop_tokens(text, stop)
        message = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async method to generate next turn in the conversation.

        Args:
            messages: The history of the conversation as a list of messages.
            stop: The list of stop words (optional).
            run_manager: The CallbackManager for LLM run, it's not used at the moment.

        Returns:
            The ChatResult that contains outputs generated by the model.

        Raises:
            ValueError: if the last message in the list is not from human.
        """
        try:
            import asyncio

            import grpc
            from google.protobuf.wrappers_pb2 import DoubleValue, Int64Value
            from yandex.cloud.ai.foundation_models.v1.foundation_models_pb2 import (
                CompletionOptions,
                Message,
            )
            from yandex.cloud.ai.foundation_models.v1.foundation_models_service_pb2 import (  # noqa: E501
                CompletionRequest,
                CompletionResponse,
            )
            from yandex.cloud.ai.foundation_models.v1.foundation_models_service_pb2_grpc import (  # noqa: E501
                TextGenerationAsyncServiceStub,
            )
            from yandex.cloud.operation.operation_service_pb2 import GetOperationRequest
            from yandex.cloud.operation.operation_service_pb2_grpc import (
                OperationServiceStub,
            )
        except ImportError as e:
            raise ImportError(
                "Please install YandexCloud SDK" " with `pip install yandexcloud`."
            ) from e
        if not messages:
            raise ValueError(
                "You should provide at least one message to start the chat!"
            )
        message_history = _parse_chat_history(messages)
        operation_api_url = "operation.api.cloud.yandex.net:443"
        channel_credentials = grpc.ssl_channel_credentials()
        async with grpc.aio.secure_channel(self.url, channel_credentials) as channel:
            request = CompletionRequest(
                model_uri=self.model_uri,
                completion_options=CompletionOptions(
                    temperature=DoubleValue(value=self.temperature),
                    max_tokens=Int64Value(value=self.max_tokens),
                ),
                messages=[Message(**message) for message in message_history],
            )
            stub = TextGenerationAsyncServiceStub(channel)
            operation = await stub.Completion(request, metadata=self._grpc_metadata)
            async with grpc.aio.secure_channel(
                operation_api_url, channel_credentials
            ) as operation_channel:
                operation_stub = OperationServiceStub(operation_channel)
                while not operation.done:
                    await asyncio.sleep(1)
                    operation_request = GetOperationRequest(operation_id=operation.id)
                    operation = await operation_stub.Get(
                        operation_request, metadata=self._grpc_metadata
                    )

            instruct_response = CompletionResponse()
            operation.response.Unpack(instruct_response)
            text = instruct_response.alternatives[0].text
            if stop is not None:
                text = enforce_stop_tokens(text, stop)
            return text
