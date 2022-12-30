"""Callback Handler that logs to streamlit."""
from typing import Any, Dict, List, Optional

import streamlit as st

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, LLMResult


class StreamlitCallbackHandler(BaseCallbackHandler):
    """Callback Handler that logs to streamlit."""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_llm_end(self, response: LLMResult) -> None:
        """Do nothing."""
        pass

    def on_llm_error(self, error: Exception) -> None:
        """Do nothing."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_chain_end(self, outputs: Dict[str, Any]) -> None:
        """Do nothing."""
        pass

    def on_chain_error(self, error: Exception) -> None:
        """Do nothing."""
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        action: AgentAction,
        color: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Print out the log in specified color."""
        # st.write requires two spaces before a newline to render it
        st.markdown(action.log.replace("\n", "  \n"))

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        st.write(f"{observation_prefix}{output}")
        st.write(llm_prefix)

    def on_tool_error(self, error: Exception) -> None:
        """Do nothing."""
        pass

    def on_agent_end(
        self, log: str, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Run when agent ends."""
        # st.write requires two spaces before a newline to render it
        st.write(log.replace("\n", "  \n"))
