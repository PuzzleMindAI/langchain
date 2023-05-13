"""Wrapper around MosaicML APIs."""
from typing import Any, Dict, List, Mapping, Optional, Tuple

import requests
from pydantic import BaseModel, Extra, root_validator

from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env


class MosaicLLMInstructorEmbeddings(BaseModel, Embeddings):
    """Wrapper around MosaicML's LLM inference service.

    To use, you should have the
    environment variable ``MOSAICML_API_TOKEN`` set with your API token, or pass
    it as a named parameter to the constructor.

    Only supports `text-generation` for now.

    Example:
        .. code-block:: python

            from langchain.llms import MosaicLLM
            endpoint_url = (
                "https://models.hosted-on.mosaicml.hosting/instructor-large/v1/predict"
            )
            mosaic_llm = MosaicLLM(
                endpoint_url=endpoint_url,
                mosaicml_api_token="my-api-key"
            )
    """

    endpoint_url: str = (
        "https://models.hosted-on.mosaicml.hosting/instructor-large/v1/predict"
    )
    """Endpoint URL to use."""
    embed_instruction: str = "Represent the document for retrieval: "
    """Instruction used to embed documents."""
    query_instruction: str = (
        "Represent the question for retrieving supporting documents: "
    )
    """Instruction used to embed the query."""

    mosaicml_api_token: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        mosaicml_api_token = get_from_dict_or_env(
            values, "mosaicml_api_token", "MOSAICML_API_TOKEN"
        )
        if not mosaicml_api_token:
            raise ValueError(
                "Could not find MOSAICML_API_TOKEN in environment or as a parameter."
            )
        values["mosaicml_api_token"] = mosaicml_api_token
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"endpoint_url": self.endpoint_url},
        }

    def _embed(self, input: List[Tuple[str, str]]) -> List[List[float]]:
        # payload samples
        parameter_payload = {"input_strings": input}

        # HTTP headers for authorization
        headers = {
            "Authorization": f"{self.mosaicml_api_token}",
            "Content-Type": "application/json",
        }

        # send request
        try:
            response = requests.post(
                self.endpoint_url, headers=headers, json=parameter_payload
            )
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        try:
            embeddings = response.json()["data"]
            if "error" in embeddings:
                raise ValueError(
                    f"Error raised by inference API: {embeddings['error']}"
                )
        except requests.exceptions.JSONDecodeError as e:
            raise ValueError(
                f"Error raised by inference API: {e}.\nResponse: {response.text}"
            )

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace instruct model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        instruction_pairs = [(self.embed_instruction, text) for text in texts]
        embeddings = self._embed(instruction_pairs)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace instruct model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        instruction_pair = (self.query_instruction, text)
        embedding = self._embed([instruction_pair])[0]
        return embedding
