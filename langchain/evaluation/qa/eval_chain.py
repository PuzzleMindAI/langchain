"""LLM Chain specifically for evaluating question answering."""
from __future__ import annotations

from typing import Any, List, Optional

from langchain.chains.llm import LLMChain
from langchain.evaluation.qa.eval_prompt import PROMPT
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate



class QAEvalChain(LLMChain):
    """LLM Chain specifically for evaluating question answering."""

    prompt: BasePromptTemplate = PROMPT
    """Prompt to use to evaluate."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, prompt=prompt, **kwargs: Any) -> QAEvalChain:
        """Load QA Eval Chain from LLM."""
        return cls(llm=llm, prompt=prompt, **kwargs)

    def evaluate(
        self,
        examples: List[dict],
        predictions: List[dict],
        question_key: str = "query",
        answer_key: str = "answer",
        prediction_key: str = "result",
    ) -> List[dict]:
        """Evaluate question answering examples and predictions."""
        inputs = []
        for i, example in enumerate(examples):
            _input = {
                "query": example[question_key],
                "answer": example[answer_key],
                "result": predictions[i][prediction_key],
            }
            inputs.append(_input)
        return self.apply(inputs)
