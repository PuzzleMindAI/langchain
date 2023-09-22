from abc import ABC
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union, cast

from langchain._api import deprecated
from langchain.pydantic_v1 import BaseModel, PrivateAttr


class BaseSerialized(TypedDict):
    """Base class for serialized objects."""

    lc: int
    id: List[str]


class SerializedConstructor(BaseSerialized):
    """Serialized constructor."""

    type: Literal["constructor"]
    kwargs: Dict[str, Any]


class SerializedSecret(BaseSerialized):
    """Serialized secret."""

    type: Literal["secret"]


class SerializedNotImplemented(BaseSerialized):
    """Serialized not implemented."""

    type: Literal["not_implemented"]
    repr: Optional[str]


class Serializable(BaseModel, ABC):
    """Serializable base class."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Is this class serializable?"""
        return False

    @property
    def lc_serializable(self) -> bool:
        """Deprecated -- instead use is_lc_serializable."""
        return self.is_lc_serializable()

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object.

        For example, if the class is `langchain.llms.openai.OpenAI`, then the
        namespace is ["langchain", "llms", "openai"]
        """
        return cls.__module__.split(".")

    @property
    def lc_namespace(self) -> List[str]:
        """Deprecated -- instead use get_lc_namespace."""
        return self.get_lc_namespace()

    @classmethod
    def get_lc_secrets(cls) -> Dict[str, str]:
        """
        Return a map of constructor argument names to secret ids.
        eg. {"openai_api_key": "OPENAI_API_KEY"}
        """
        return dict()

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """Deprecated -- instead use get_lc_secrets."""
        return self.get_lc_secrets()

    def get_lc_attributes(self) -> Dict[str, Any]:
        """
        Return a list of attribute names that should be included in the
        serialized kwargs. These attributes must be accepted by the
        constructor.
        """
        return {}

    @property
    def lc_attributes(self) -> Dict:
        """Deprecated -- instead use get_lc_attributes."""
        return self.get_lc_attributes()

    class Config:
        extra = "ignore"

    _lc_kwargs = PrivateAttr(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lc_kwargs = kwargs

    def to_json(self) -> Union[SerializedConstructor, SerializedNotImplemented]:
        if not self.lc_serializable:
            return self.to_json_not_implemented()

        secrets = dict()
        # Get latest values for kwargs if there is an attribute with same name
        lc_kwargs = {
            k: getattr(self, k, v)
            for k, v in self._lc_kwargs.items()
            if not (self.__exclude_fields__ or {}).get(k, False)  # type: ignore
        }

        # Merge the lc_secrets and lc_attributes from every class in the MRO
        for cls in [None, *self.__class__.mro()]:
            # Once we get to Serializable, we're done
            if cls is Serializable:
                break

            # Get a reference to self bound to each class in the MRO
            this = cast(Serializable, self if cls is None else super(cls, self))

            if this.lc_secrets != Serializable.lc_secrets:
                lc_secrets = this.lc_secrets
            else:
                lc_secrets = this.get_lc_secrets()

            secrets.update(lc_secrets)

            if this.lc_attributes != Serializable.lc_attributes:
                lc_kwargs.update(this.lc_attributes)
            else:
                lc_kwargs.update(this.get_lc_attributes())

        # include all secrets, even if not specified in kwargs
        # as these secrets may be passed as an environment variable instead
        for key in secrets.keys():
            secret_value = getattr(self, key, None) or lc_kwargs.get(key)
            if secret_value is not None:
                lc_kwargs.update({key: secret_value})

        if type(self).lc_namespace != Serializable.lc_namespace:
            namespace = self.lc_namespace
        else:
            namespace = self.get_lc_namespace()

        return {
            "lc": 1,
            "type": "constructor",
            "id": [*namespace, self.__class__.__name__],
            "kwargs": lc_kwargs
            if not secrets
            else _replace_secrets(lc_kwargs, secrets),
        }

    def to_json_not_implemented(self) -> SerializedNotImplemented:
        return to_json_not_implemented(self)


def _replace_secrets(
    root: Dict[Any, Any], secrets_map: Dict[str, str]
) -> Dict[Any, Any]:
    result = root.copy()
    for path, secret_id in secrets_map.items():
        [*parts, last] = path.split(".")
        current = result
        for part in parts:
            if part not in current:
                break
            current[part] = current[part].copy()
            current = current[part]
        if last in current:
            current[last] = {
                "lc": 1,
                "type": "secret",
                "id": [secret_id],
            }
    return result


def to_json_not_implemented(obj: object) -> SerializedNotImplemented:
    """Serialize a "not implemented" object.

    Args:
        obj: object to serialize

    Returns:
        SerializedNotImplemented
    """
    _id: List[str] = []
    try:
        if hasattr(obj, "__name__"):
            _id = [*obj.__module__.split("."), obj.__name__]
        elif hasattr(obj, "__class__"):
            _id = [*obj.__class__.__module__.split("."), obj.__class__.__name__]
    except Exception:
        pass

    result: SerializedNotImplemented = {
        "lc": 1,
        "type": "not_implemented",
        "id": _id,
        "repr": None,
    }
    try:
        result["repr"] = repr(obj)
    except Exception:
        pass
    return result
