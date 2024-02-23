import copy
import re
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from langchain_community.utils.math import (
    cosine_similarity,
)
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.embeddings import Embeddings


def combine_sentences(sentences: List[dict], buffer_size: int = 1) -> List[dict]:
    # Go through each sentence dict
    for i in range(len(sentences)):
        # Create a string that will hold the sentences which are joined
        combined_sentence = ""

        # Add sentences before the current one, based on the buffer size.
        for j in range(i - buffer_size, i):
            # Check if the index j is not negative
            # (to avoid index out of range like on the first one)
            if j >= 0:
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += sentences[j]["sentence"] + " "

        # Add the current sentence
        combined_sentence += sentences[i]["sentence"]

        # Add sentences after the current one, based on the buffer size
        for j in range(i + 1, i + 1 + buffer_size):
            # Check if the index j is within the range of the sentences list
            if j < len(sentences):
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += " " + sentences[j]["sentence"]

        # Then add the whole thing to your dict
        # Store the combined sentence in the current sentence dict
        sentences[i]["combined_sentence"] = combined_sentence

    return sentences


def calculate_cosine_distances(sentences: List[dict]) -> Tuple[List[float], List[dict]]:
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]["combined_sentence_embedding"]
        embedding_next = sentences[i + 1]["combined_sentence_embedding"]

        # Calculate cosine similarity
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]

        # Convert to cosine distance
        distance = 1 - similarity

        # Append cosine distance to the list
        distances.append(distance)

        # Store distance in the dictionary
        sentences[i]["distance_to_next"] = distance

    # Optionally handle the last sentence
    # sentences[-1]['distance_to_next'] = None  # or a default value

    return distances, sentences


def split_string(input_str, limit, sep=" "):
    # Split the input string into words
    words = input_str.split()

    # Check if any single word exceeds the limit, which is not allowed
    if max(map(len, words)) > limit:
        raise ValueError("A single word exceeds the limit, making splitting impossible.")

    # Initialize the result list, the current part being constructed, and the remaining words
    res = []  # List to store the final result of split parts
    part = words[0]  # Start the first part with the first word
    others = words[1:]  # Remaining words to process

    # Iterate through the remaining words
    for word in others:
        # Check if adding the next word exceeds the limit for the current part
        if len(sep) + len(word) > limit - len(part):
            # If it does, add the current part to the result list and start a new part
            res.append(part)
            part = word
        else:
            # Otherwise, add the word to the current part
            part += sep + word

    # After the loop, add the last part to the result list if it's not empty
    if part:
        res.append(part)

    return res


def add_chunk(sentences, start_index, end_index, chunks, max_chunk_size):
    """Adds sentences as a chunk if their total length does not exceed max_chunk_size."""
    if not max_chunk_size:
        combined_text = " ".join([d["sentence"] for d in sentences[start_index : end_index]])
        chunks.append(combined_text)
    else:
        group = []
        chunk_size = 0
        for i in range(start_index, end_index):
            sentence_length = len(sentences[i]["sentence"])
            if chunk_size + sentence_length > max_chunk_size:
                if group:  # Ensures that the group is not empty
                    chunks.append(" ".join([d["sentence"] for d in group]))
                group = [sentences[i]]  # Start a new group with the current sentence
                chunk_size = sentence_length  # Resets chunk size
            else:
                group.append(sentences[i])
                chunk_size += sentence_length

        # Adds the last group if it is not empty
        if group:
            chunks.append(" ".join([d["sentence"] for d in group]))


class SemanticChunker(BaseDocumentTransformer):
    """Splits the text based on semantic similarity.

    Taken from Greg Kamradt's wonderful notebook:
    https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/5_Levels_Of_Text_Splitting.ipynb

    All credit to him.

    At a high level, this splits into sentences, then groups into groups of 3
    sentences, and then merges one that are similar in the embedding space.
    """

    def __init__(self, embeddings: Embeddings, add_start_index: bool = False, max_chunk_size: int = None):
        self._add_start_index = add_start_index
        self.embeddings = embeddings
        self.max_chunk_size = max_chunk_size

    def split_text(self, text: str) -> List[str]:
        """Split text into multiple components."""
        # Splitting the essay on '.', '?', and '!'
        single_sentences_list = re.split(r"(?<=[.?!])\s+", text)

        # having len(single_sentences_list) == 1 would cause the following
        # np.percentile to fail.
        if len(single_sentences_list) == 1:
            return single_sentences_list

        if self.max_chunk_size:

            # Preparing a new list to store the results
            new_single_sentences_list = []

            for sentence in single_sentences_list:
                # Check whether the sentence exceeds the maximum authorised size
                if len(sentence) >= self.max_chunk_size:
                    # Dividing the sentence into sub-parts
                    sentences = split_string(sentence, self.max_chunk_size, " ")
                    # Extension of the new list by sub-parts
                    new_single_sentences_list.extend(sentences)
                else:
                    # Add the original sentence if it does not exceed the maximum size
                    new_single_sentences_list.append(sentence)

            # Replacing the original list with the new one
            single_sentences_list = new_single_sentences_list

        sentences = [
            {"sentence": x, "index": i} for i, x in enumerate(single_sentences_list)
        ]
        sentences = combine_sentences(sentences)
        embeddings = self.embeddings.embed_documents(
            [x["combined_sentence"] for x in sentences]
        )
        for i, sentence in enumerate(sentences):
            sentence["combined_sentence_embedding"] = embeddings[i]
        distances, sentences = calculate_cosine_distances(sentences)
        start_index = 0

        # Create a list to hold the grouped sentences
        chunks = []
        breakpoint_percentile_threshold = 95
        breakpoint_distance_threshold = np.percentile(
            distances, breakpoint_percentile_threshold
        )  # If you want more chunks, lower the percentile cutoff

        indices_above_thresh = [
            i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
        ]  # The indices of those breakpoints on your list

        # Iterate through the breakpoints to slice the sentences
        for index in indices_above_thresh:
            # The end index is the current breakpoint
            end_index = index

            # Slice the sentence_dicts from the current start index to the end index
            add_chunk(sentences, start_index, end_index + 1, chunks, self.max_chunk_size)

            # Update the start index for the next group
            start_index = index + 1

        # The last group, if any sentences remain
        if start_index < len(sentences):
            add_chunk(sentences, start_index, len(sentences), chunks, self.max_chunk_size)

        return chunks

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            index = -1
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])
                if self._add_start_index:
                    index = text.find(chunk, index + 1)
                    metadata["start_index"] = index
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Split documents."""
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Transform sequence of documents by splitting them."""
        return self.split_documents(list(documents))
