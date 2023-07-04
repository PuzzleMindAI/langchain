"""Toolkit for interacting with a Power BI dataset."""
from typing import List, Optional

from pydantic import Field

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain.tools.powerbi.prompt import (
    SINGLE_QUESTION_TO_QUERY,
    QUESTION_TO_QUERY_BASE,
    USER_INPUT,
)
from langchain.tools.powerbi.tool import (
    InfoPowerBITool,
    ListPowerBITool,
    QueryPowerBITool,
)
from langchain.utilities.powerbi import PowerBIDataset


class PowerBIToolkit(BaseToolkit):
    """Toolkit for interacting with PowerBI dataset."""

    powerbi: PowerBIDataset = Field(exclude=True)
    llm: BaseLanguageModel | BaseChatModel = Field(exclude=True)
    examples: Optional[str] = None
    max_iterations: int = 5
    callback_manager: Optional[BaseCallbackManager] = None
    output_token_limit: Optional[int] = None
    tiktoken_model_name: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            QueryPowerBITool(
                llm_chain=self._get_chain(),
                powerbi=self.powerbi,
                examples=self.examples,
                max_iterations=self.max_iterations,
                output_token_limit=self.output_token_limit,
                tiktoken_model_name=self.tiktoken_model_name,
            ),
            InfoPowerBITool(powerbi=self.powerbi),
            ListPowerBITool(powerbi=self.powerbi),
        ]

    def _get_chain(self) -> LLMChain:
        """Construct the chain based on the callback manager and model type."""
        if isinstance(self.llm, BaseChatModel):
            system_prompt = SystemMessagePromptTemplate(
                prompt=PromptTemplate(
                    template=QUESTION_TO_QUERY_BASE,
                    input_variables=["tables", "schemas", "examples"],
                )
            )
            human_prompt = HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    template=USER_INPUT,
                    input_variables=["tool_input"],
                )
            )
            prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
        else:
            prompt = PromptTemplate(
                template=SINGLE_QUESTION_TO_QUERY,
                input_variables=["tool_input", "tables", "schemas", "examples"],
            )
        return LLMChain(
            llm=self.llm,
            callback_manager=self.callback_manager if self.callback_manager else None,
            prompt=prompt,
        )
