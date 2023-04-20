import pytest

from langchain.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper


def ddg_installed() -> bool:
    try:
        from duckduckgo_search import ddg  # noqa: F401

        return True
    except Exception as e:
        print(f"duckduckgo not installed, skipping test {e}")
        return False


@pytest.mark.skipif(not ddg_installed(), reason="requires duckduckgo-search package")
def test_ddg_search_tool() -> None:
    keywords = "Bella Ciao"
    tool = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper())
    result = tool(keywords)
    print(result)
    assert len(result.split()) > 20
