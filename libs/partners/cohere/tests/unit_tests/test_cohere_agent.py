from typing import Any, Dict, Optional, Type, Union

import pytest
from langchain_core.tools import BaseModel, BaseTool, Field

from langchain_cohere.cohere_agent import (
    _format_to_cohere_tools,
    _remove_signature_from_description,
)

expected_test_tool_definition = {
    "description": "test_tool description",
    "name": "test_tool",
    "parameter_definitions": {
        "arg_1": {
            "description": "Arg1 description",
            "required": True,
            "type": "str",
        },
        "optional_arg_2": {
            "description": "Arg2 description",
            "required": False,
            "type": "str",
        },
        "arg_3": {
            "description": "Arg3 description",
            "required": True,
            "type": "int",
        },
    },
}


class _TestToolSchema(BaseModel):
    arg_1: str = Field(description="Arg1 description")
    optional_arg_2: Optional[str] = Field(description="Arg2 description", default="2")
    arg_3: int = Field(description="Arg3 description")


class _TestTool(BaseTool):
    name = "test_tool"
    description = "test_tool description"
    args_schema: Type[_TestToolSchema] = _TestToolSchema

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        pass


class test_tool(BaseModel):
    """test_tool description"""

    arg_1: str = Field(description="Arg1 description")
    optional_arg_2: Optional[str] = Field(description="Arg2 description", default="2")
    arg_3: int = Field(description="Arg3 description")


test_tool_as_dict = {
    "title": "test_tool",
    "description": "test_tool description",
    "properties": {
        "arg_1": {"description": "Arg1 description", "type": "string"},
        "optional_arg_2": {
            "description": "Arg2 description",
            "type": "string",
            "default": "2",
        },
        "arg_3": {"description": "Arg3 description", "type": "integer"},
    },
}


@pytest.mark.parametrize(
    "tool",
    [
        pytest.param(_TestTool(), id="tool from BaseTool"),
        pytest.param(test_tool, id="BaseModel"),
        pytest.param(test_tool_as_dict, id="JSON schema dict"),
    ],
)
def test_format_to_cohere_tools(
    tool: Union[Dict[str, Any], BaseTool, Type[BaseModel]],
) -> None:
    actual = _format_to_cohere_tools([tool])

    assert [expected_test_tool_definition] == actual


@pytest.mark.parametrize(
    "name,description,expected",
    [
        pytest.param(
            "foo", "bar baz", "bar baz", id="description doesn't have signature"
        ),
        pytest.param("foo", "", "", id="description is empty"),
        pytest.param("foo", "foo(a: str) - bar baz", "bar baz", id="signature"),
        pytest.param(
            "foo", "foo() - bar baz", "bar baz", id="signature with empty args"
        ),
        pytest.param(
            "foo",
            "foo(a: str) - foo(b: str) - bar",
            "foo(b: str) - bar",
            id="signature with edge case",
        ),
        pytest.param(
            "foo", "foo() -> None - bar baz", "bar baz", id="signature with return type"
        ),
    ],
)
def test_remove_signature_from_description(
    name: str, description: str, expected: str
) -> None:
    actual = _remove_signature_from_description(name=name, description=description)

    assert expected == actual
