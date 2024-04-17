"""Agent for working with polars objects."""
from typing import Any, Dict, List, Optional, Tuple

from langchain.agents.agent import AgentExecutor, BaseSingleActionAgent
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents.types import AgentType
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.schema import BasePromptTemplate, SystemMessage
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools.python.tool import PythonAstREPLTool

from langchain_experimental.dataframe_agents.pandas.prompt import (
    FUNCTIONS_WITH_DF,
    FUNCTIONS_WITH_MULTI_DF,
    MULTI_DF_PREFIX,
    MULTI_DF_PREFIX_FUNCTIONS,
    PREFIX,
    PREFIX_FUNCTIONS,
    SUFFIX_NO_DF,
    SUFFIX_WITH_DF,
    SUFFIX_WITH_MULTI_DF,
)

PREFIX = PREFIX.format(lib="polars")
MULTI_DF_PREFIX = MULTI_DF_PREFIX.format(lib="polars", num_dfs="{num_dfs}")
PREFIX_FUNCTIONS = PREFIX_FUNCTIONS.format(lib="polars")
MULTI_DF_PREFIX_FUNCTIONS = MULTI_DF_PREFIX_FUNCTIONS.format(
    lib="polars", num_dfs="{num_dfs}"
)


def _convert_pl_dataframe_to_markdown_table(df: Any) -> str:
    markdown_table = (
        "| "
        + " | ".join(df.columns)
        + " |\n"
        + "| "
        + " | ".join(["---"] * len(df.columns))
        + " |\n"
    )

    for row in df.rows():
        markdown_table += "| " + " | ".join(str(cell) for cell in row) + " |\n"

    return markdown_table


def _get_multi_prompt(
    dfs: List[Any],
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
) -> Tuple[BasePromptTemplate, List[PythonAstREPLTool]]:
    num_dfs = len(dfs)
    if suffix is not None:
        suffix_to_use = suffix
        include_dfs_head = True
    elif include_df_in_prompt:
        suffix_to_use = SUFFIX_WITH_MULTI_DF
        include_dfs_head = True
    else:
        suffix_to_use = SUFFIX_NO_DF
        include_dfs_head = False
    if input_variables is None:
        input_variables = ["input", "agent_scratchpad", "num_dfs"]
        if include_dfs_head:
            input_variables += ["dfs_head"]

    if prefix is None:
        prefix = MULTI_DF_PREFIX

    df_locals = {}
    for i, dataframe in enumerate(dfs):
        df_locals[f"df{i + 1}"] = dataframe
    tools = [PythonAstREPLTool(locals=df_locals)]

    prompt = ZeroShotAgent.create_prompt(
        tools, prefix=prefix, suffix=suffix_to_use, input_variables=input_variables
    )

    partial_prompt = prompt.partial()
    if "dfs_head" in input_variables:
        dfs_head = "\n\n".join(
            [
                _convert_pl_dataframe_to_markdown_table(d.head(number_of_head_rows))
                for d in dfs
            ]
        )
        partial_prompt = partial_prompt.partial(num_dfs=str(num_dfs), dfs_head=dfs_head)
    if "num_dfs" in input_variables:
        partial_prompt = partial_prompt.partial(num_dfs=str(num_dfs))
    return partial_prompt, tools


def _get_single_prompt(
    df: Any,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
) -> Tuple[BasePromptTemplate, List[PythonAstREPLTool]]:
    if suffix is not None:
        suffix_to_use = suffix
        include_df_head = True
    elif include_df_in_prompt:
        suffix_to_use = SUFFIX_WITH_DF
        include_df_head = True
    else:
        suffix_to_use = SUFFIX_NO_DF
        include_df_head = False

    if input_variables is None:
        input_variables = ["input", "agent_scratchpad"]
        if include_df_head:
            input_variables += ["df_head"]

    if prefix is None:
        prefix = PREFIX

    tools = [PythonAstREPLTool(locals={"df": df})]

    prompt = ZeroShotAgent.create_prompt(
        tools, prefix=prefix, suffix=suffix_to_use, input_variables=input_variables
    )

    partial_prompt = prompt.partial()
    if "df_head" in input_variables:
        partial_prompt = partial_prompt.partial(
            df_head=_convert_pl_dataframe_to_markdown_table(
                df.head(number_of_head_rows)
            )
        )
    return partial_prompt, tools


def _get_prompt_and_tools(
    df: Any,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
) -> Tuple[BasePromptTemplate, List[PythonAstREPLTool]]:
    try:
        import sys

        import polars as pl

        pl.Config(fmt_str_lengths=sys.maxsize)
    except ImportError:
        raise ImportError(
            "polars package not found, please install with `pip install polars`"
        )

    if include_df_in_prompt is not None and suffix is not None:
        raise ValueError("If suffix is specified, include_df_in_prompt should not be.")

    if isinstance(df, list):
        for item in df:
            if not isinstance(item, pl.DataFrame):
                raise ValueError(f"Expected polars object, got {type(df)}")
        return _get_multi_prompt(
            df,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
            include_df_in_prompt=include_df_in_prompt,
            number_of_head_rows=number_of_head_rows,
        )
    else:
        if not isinstance(df, pl.DataFrame):
            raise ValueError(f"Expected polars object, got {type(df)}")
        return _get_single_prompt(
            df,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
            include_df_in_prompt=include_df_in_prompt,
            number_of_head_rows=number_of_head_rows,
        )


def _get_functions_single_prompt(
    df: Any,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
) -> Tuple[BasePromptTemplate, List[PythonAstREPLTool]]:
    if suffix is not None:
        suffix_to_use = suffix
        if include_df_in_prompt:
            suffix_to_use = suffix_to_use.format(
                df_head=_convert_pl_dataframe_to_markdown_table(
                    df.head(number_of_head_rows)
                )
            )
    elif include_df_in_prompt:
        suffix_to_use = FUNCTIONS_WITH_DF.format(
            df_head=_convert_pl_dataframe_to_markdown_table(
                df.head(number_of_head_rows)
            )
        )
    else:
        suffix_to_use = ""

    if prefix is None:
        prefix = PREFIX_FUNCTIONS

    tools = [PythonAstREPLTool(locals={"df": df})]
    system_message = SystemMessage(content=prefix + suffix_to_use)
    prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
    return prompt, tools


def _get_functions_multi_prompt(
    dfs: Any,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
) -> Tuple[BasePromptTemplate, List[PythonAstREPLTool]]:
    if suffix is not None:
        suffix_to_use = suffix
        if include_df_in_prompt:
            dfs_head = "\n\n".join(
                [
                    _convert_pl_dataframe_to_markdown_table(d.head(number_of_head_rows))
                    for d in dfs
                ]
            )
            suffix_to_use = suffix_to_use.format(
                dfs_head=dfs_head,
            )
    elif include_df_in_prompt:
        dfs_head = "\n\n".join(
            [
                _convert_pl_dataframe_to_markdown_table(d.head(number_of_head_rows))
                for d in dfs
            ]
        )
        suffix_to_use = FUNCTIONS_WITH_MULTI_DF.format(
            dfs_head=dfs_head,
        )
    else:
        suffix_to_use = ""

    if prefix is None:
        prefix = MULTI_DF_PREFIX_FUNCTIONS
    prefix = prefix.format(num_dfs=str(len(dfs)))

    df_locals = {}
    for i, dataframe in enumerate(dfs):
        df_locals[f"df{i + 1}"] = dataframe
    tools = [PythonAstREPLTool(locals=df_locals)]
    system_message = SystemMessage(content=prefix + suffix_to_use)
    prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
    return prompt, tools


def _get_functions_prompt_and_tools(
    df: Any,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
) -> Tuple[BasePromptTemplate, List[PythonAstREPLTool]]:
    try:
        import sys

        import polars as pl

        pl.Config(fmt_str_lengths=sys.maxsize)
    except ImportError:
        raise ImportError(
            "polars package not found, please install with `pip install polars`"
        )
    if input_variables is not None:
        raise ValueError("`input_variables` is not supported at the moment.")

    if include_df_in_prompt is not None and suffix is not None:
        raise ValueError("If suffix is specified, include_df_in_prompt should not be.")

    if isinstance(df, list):
        for item in df:
            if not isinstance(item, pl.DataFrame):
                raise ValueError(f"Expected polars object, got {type(df)}")
        return _get_functions_multi_prompt(
            df,
            prefix=prefix,
            suffix=suffix,
            include_df_in_prompt=include_df_in_prompt,
            number_of_head_rows=number_of_head_rows,
        )
    else:
        if not isinstance(df, pl.DataFrame):
            raise ValueError(f"Expected polars object, got {type(df)}")
        return _get_functions_single_prompt(
            df,
            prefix=prefix,
            suffix=suffix,
            include_df_in_prompt=include_df_in_prompt,
            number_of_head_rows=number_of_head_rows,
        )


def create_polars_dataframe_agent(
    llm: BaseLanguageModel,
    df: Any,
    agent_type: AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    verbose: bool = False,
    return_intermediate_steps: bool = False,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
    **kwargs: Dict[str, Any],
) -> AgentExecutor:
    """Construct a polars agent from an LLM and dataframe."""
    agent: BaseSingleActionAgent
    if agent_type == AgentType.ZERO_SHOT_REACT_DESCRIPTION:
        prompt, tools = _get_prompt_and_tools(
            df,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
            include_df_in_prompt=include_df_in_prompt,
            number_of_head_rows=number_of_head_rows,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        agent = ZeroShotAgent(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            callback_manager=callback_manager,
            **kwargs,
        )
    elif agent_type == AgentType.OPENAI_FUNCTIONS:
        _prompt, tools = _get_functions_prompt_and_tools(
            df,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
            include_df_in_prompt=include_df_in_prompt,
            number_of_head_rows=number_of_head_rows,
        )
        agent = OpenAIFunctionsAgent(
            llm=llm,
            prompt=_prompt,
            tools=tools,
            callback_manager=callback_manager,
            **kwargs,
        )
    else:
        raise ValueError(f"Agent type {agent_type} not supported at the moment.")
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        return_intermediate_steps=return_intermediate_steps,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
        **(agent_executor_kwargs or {}),
    )
