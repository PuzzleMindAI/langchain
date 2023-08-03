from typing import Any, Dict, List, Literal

from pydantic.class_validators import root_validator
from pydantic.config import Extra

from typing import Optional
from langchain.schema import BaseRetriever, Document

from ..utilities.google_drive import (
    GoogleDriveUtilities,
    get_template,
)
from langchain.callbacks.manager import Callbacks


class GoogleDriveRetriever(GoogleDriveUtilities, BaseRetriever):
    """Wrapper around Google Drive API.

    The application must be authenticated with a json file.
    The format may be for a user or for an application via a service account.
    The environment variable `GOOGLE_ACCOUNT_FILE` may be set to reference this file.
    For more information, see [here]
    (https://developers.google.com/workspace/guides/auth-overview).
    """

    class Config:
        extra = Extra.allow
        allow_mutation = False
        underscore_attrs_are_private = True

    mode: Literal[
        "snippets", "snippets-markdown", "documents", "documents-markdown"
    ] = "snippets-markdown"

    @root_validator(pre=True)
    def validate_template(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        folder_id = v.get("folder_id")

        if not v.get("template"):
            if folder_id:
                template = get_template("gdrive-query-in-folders")
            else:
                template = get_template("gdrive-query")
            v["template"] = template
        return v

    def get_relevant_documents(
        self,
        query: str,
        *,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        return list(
            self.lazy_get_relevant_documents(
                query=query,
                callbacks=callbacks,
                tags=tags,
                metadata=metadata,
                **kwargs,
            )
        )

    async def aget_relevant_documents(
        self,
        query: str,
        *,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Get documents relevant for a query.

        NOT IMPLEMENTED

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        raise NotImplementedError("GoogleSearchRun does not support async")
