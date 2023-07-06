import json
from typing import Any, List, Union, Dict, Optional

from pydantic import BaseModel, root_validator, validator

from langchain.schema import BaseLLMOutputParser, ChatGeneration, Generation


class OutputFunctionsParser(BaseLLMOutputParser[Any]):
    args_only: bool = False

    def parse_result(self, result: List[Generation]) -> Any:
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            raise ValueError(
                "This output parser can only be used with a chat generation."
            )
        message = generation.message
        try:
            func_call = message.additional_kwargs["function_call"]
        except ValueError as exc:
            raise ValueError(f"Could not parse function call: {exc}")

        if self.args_only:
            return func_call["arguments"]
        return func_call


class JsonOutputFunctionsParser(OutputFunctionsParser):
    def parse_result(self, result: List[Generation]) -> Any:
        func = super().parse_result(result)
        if self.args_only:
            return json.loads(func)
        func["arguments"] = json.loads(func["arguments"])
        return func


class JsonKeyOutputFunctionsParser(JsonOutputFunctionsParser):
    key_name: str

    def parse_result(self, result: List[Generation]) -> Any:
        res = super().parse_result(result)
        return res[self.key_name]


class PydanticOutputFunctionsParser(OutputFunctionsParser):
    pydantic_schema: Union[BaseModel, Dict[str, BaseModel]]

    @root_validator(pre=True)
    def validate_schema(self, values: Dict) -> Dict:
        schema = values["pydantic_schema"]
        if "args_only" not in values:
            values["args_only"] = isinstance(schema, BaseModel)
        elif values["args_only"] and isinstance(schema, Dict):
            raise ValueError(
                "If multiple pydantic schemas are provided then args_only should be"
                " False."
            )
        return values

    def parse_result(self, result: List[Generation]) -> Any:
        _result = super().parse_result(result)
        if self.args_only:
            pydantic_args = self.pydantic_schema.parse_raw(_result)
        else:
            fn_name = _result["name"]
            _args = _result["arguments"]
            pydantic_args = self.pydantic_schema[fn_name].parse_raw(_args)
        return pydantic_args


class PydanticAttrOutputFunctionsParser(PydanticOutputFunctionsParser):
    attr_name: str

    def parse_result(self, result: List[Generation]) -> Any:
        result = super().parse_result(result)
        return getattr(result, self.attr_name)
