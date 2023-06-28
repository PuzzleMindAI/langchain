from typing import Dict, Iterator, List, Union

import requests

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.parsers.grobid import GrobidParser


class ServerUnavailableException(Exception):
    pass


class GrobidLoader(BaseLoader):
    """Loader that uses Grobid to load article PDF files."""

    def __init__(
        self,
        file_path: str,
        segment_sentences: bool,
        grobid_server: str = "http://localhost:8070/api/processFulltextDocument",
    ) -> None:
        try:
            r = requests.get(grobid_server)
        except requests.exceptions.RequestException:
            print(
                "GROBID server does not appear up and running, \
                 please ensure Grobid is installed and the server is running"
            )
            raise ServerUnavailableException
        pdf = open(file_path, "rb")
        files = {"input": (file_path, pdf, "application/pdf", {"Expires": "0"})}
        try:
            data: Dict[str, Union[str, List[str]]] = {}
            for param in ["generateIDs", "consolidateHeader", "segmentSentences"]:
                data[param] = "1"
            data["teiCoordinates"] = ["head", "s"]

            # headers = {"Accept": "application/xml"}
            files = files or {}
            r = requests.request(
                "POST",
                grobid_server,
                headers=None,
                params=None,
                files=files,
                data=data,
                timeout=60,
            )
            # xml_data, status = r.text, r.status_code
            xml_data = r.text
        except requests.exceptions.ReadTimeout:
            status, xml_data = 408, None

        self.file_path = file_path
        self.xml_data = xml_data
        self.segment_sentences = segment_sentences

    def lazy_load(self) -> Iterator[Document]:
        """Lazy_load file."""
        parser = GrobidParser()
        if self.xml_data is None:
            return iter([])
        assert self.xml_data is not None
        return parser.lazy_parse(self.file_path, self.xml_data, self.segment_sentences)

    def load(self) -> List[Document]:
        """Load file."""
        return list(self.lazy_load())
