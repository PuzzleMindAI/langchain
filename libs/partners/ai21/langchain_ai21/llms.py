import asyncio
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Iterator,
    List,
    Optional,
)

from ai21.models import CompletionsResponse, Penalty

from langchain_ai21.ai21_base import AI21Base
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import GenerationChunk, LLMResult, Generation, RunInfo


class AI21LLM(BaseLLM, AI21Base):
    """AI21LLM large language models.

    Example:
        .. code-block:: python

            from langchain_ai21 import AI21LLM

            model = AI21LLM()
    """

    model: str = "j2-ultra"
    max_tokens: Optional[int] = None
    num_results: Optional[int] = None
    min_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k_returns: Optional[int] = None
    custom_model: Optional[str] = None
    frequency_penalty: Optional[Penalty] = None
    presence_penalty: Optional[Penalty] = None
    count_penalty: Optional[Penalty] = None
    epoch: Optional[int] = None

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "ai21-llm"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations: List[List[Generation]] = []

        for prompt in prompts:
            response = self._invoke_completion(
                prompt=prompt, model=self.model, stop_sequences=stop, **kwargs
            )
            generation = self._response_to_generation(response)
            generations.append(generation)

        return LLMResult(generations=generations)

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        # Change implementation if integration natively supports async generation.
        return await asyncio.get_running_loop().run_in_executor(
            None, partial(self._generate, **kwargs), prompts, stop, run_manager
        )

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        raise NotImplementedError

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        yield GenerationChunk(text="Yield chunks")
        yield GenerationChunk(text=" like this!")

    def _invoke_completion(
        self,
        prompt: str,
        model: str,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> CompletionsResponse:
        return self.client.completion.create(
            prompt=prompt,
            model=model,
            max_tokens=self.max_tokens,
            num_results=self.num_results,
            min_tokens=self.min_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k_return=self.top_k_returns,
            custom_model=self.custom_model,
            stop_sequences=stop_sequences,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            count_penalty=self.count_penalty,
            epoch=self.epoch,
        )

    def _response_to_generation(
        self, response: CompletionsResponse
    ) -> List[Generation]:
        return [
            Generation(
                text=completion.data.text,
                generation_info=completion.to_dict(),
            )
            for completion in response.completions
        ]
