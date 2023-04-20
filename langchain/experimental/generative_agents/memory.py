import math
import faiss
import re
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from termcolor import colored

from pydantic import BaseModel, Field

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import BaseLanguageModel, Document, BaseMemory
from langchain.vectorstores import FAISS

from typing import Dict, Any, List, Optional, Tuple


class GenerativeAgentsMemory(BaseMemory):

    llm: BaseLanguageModel

    memory_retriever: TimeWeightedVectorStoreRetriever
    """The retriever to fetch related memories."""
    verbose: bool = False
    reflection_threshold: Optional[float] = None
    """When the total 'importance' of memories exceeds the above threshold, stop to reflect."""

    current_plan: List[str] = []
    """The current plan of the agent."""

    memory_importance: float = 0.0  # : :meta private:
    max_tokens_limit: int = 1200  # : :meta private:
    queries_key: str = "queries"
    answer_key: str = "relevant_memories"
    add_key: str = "add"

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse a newline-separated string into a list of strings."""
        lines = re.split(r'\n', text.strip())
        return [re.sub(r'^\s*\d+\.\s*', '', line).strip() for line in lines]

    def _get_topics_of_reflection(self, last_k: int = 50) -> Tuple[str, str, str]:
        """Return the 3 most salient high-level questions about recent observations."""
        prompt = PromptTemplate.from_template(
            "{observations}\n\n"
            + "Given only the information above, what are the 3 most salient"
            + " high-level questions we can answer about the subjects in the statements?"
            + " Provide each question on a new line.\n\n"
        )
        reflection_chain = LLMChain(
            llm=self.llm, prompt=prompt, verbose=self.verbose)
        observations = self.memory_retriever.memory_stream[-last_k:]
        observation_str = "\n".join([o.page_content for o in observations])
        result = reflection_chain.run(observations=observation_str)
        return self._parse_list(result)

    def _get_insights_on_topic(self, topic: str) -> List[str]:
        """Generate 'insights' on a topic of reflection, based on pertinent memories."""
        prompt = PromptTemplate.from_template(
            "Statements about {topic}\n"
            + "{related_statements}\n\n"
            + "What 5 high-level insights can you infer from the above statements?"
            + " (example format: insight (because of 1, 5, 3))"
        )
        related_memories = self.fetch_memories(topic)
        related_statements = "\n".join([f"{i+1}. {memory.page_content}"
                                        for i, memory in
                                        enumerate(related_memories)])
        reflection_chain = LLMChain(
            llm=self.llm, prompt=prompt, verbose=self.verbose)
        result = reflection_chain.run(
            topic=topic, related_statements=related_statements)
        # TODO: Parse the connections between memories and insights
        return self._parse_list(result)

    def pause_to_reflect(self) -> List[str]:
        """Reflect on recent observations and generate 'insights'."""
        print(colored(f"Character is reflecting", "blue"))
        new_insights = []
        topics = self._get_topics_of_reflection()
        for topic in topics:
            insights = self._get_insights_on_topic(topic)
            for insight in insights:
                self.add_memory(insight)
            new_insights.extend(insights)
        return new_insights

    def _score_memory_importance(self, memory_content: str, weight: float = 0.15) -> float:
        """Score the absolute importance of the given memory."""
        # A weight of 0.25 makes this less important than it
        # would be otherwise, relative to salience and time
        prompt = PromptTemplate.from_template(
            "On the scale of 1 to 10, where 1 is purely mundane"
            + " (e.g., brushing teeth, making bed) and 10 is"
            + " extremely poignant (e.g., a break up, college"
            + " acceptance), rate the likely poignancy of the"
            + " following piece of memory. Respond with a single integer."
            + "\nMemory: {memory_content}"
            + "\nRating: "
        )
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        score = chain.run(memory_content=memory_content).strip()
        # TODO: use better log function
        if self.verbose:
            print(colored(f"Importance score: {score}", "blue"))
        match = re.search(r"^\D*(\d+)", score)
        if match:
            return (float(score[0]) / 10) * weight
        else:
            return 0.0

    def add_memory(self, memory_content: str) -> List[str]:
        """Add an observation or memory to the agent's memory."""
        importance_score = self._score_memory_importance(memory_content)
        self.memory_importance += importance_score
        document = Document(page_content=memory_content, metadata={
                            "importance": importance_score})
        result = self.memory_retriever.add_documents([document])

        # After an agent has processed a certain amount of memories (as measured by
        # aggregate importance), it is time to reflect on recent events to add
        # more synthesized memories to the agent's memory stream.
        if (self.reflection_threshold is not None
                and self.memory_importance > self.reflection_threshold):
            self.pause_to_reflect()
            # Hack to clear the importance from reflection
            self.memory_importance = 0.0
        return result

    def fetch_memories(self, observation: str) -> List[Document]:
        """Fetch related memories."""
        return self.memory_retriever.get_relevant_documents(observation)

    @property
    def memory_variables(self) -> List[str]:
        """Input keys this memory class will load dynamically."""
        return [self.answer_key]

    def load_memory_variables(self, inputs: Dict[str, List[str]]) -> Dict[str, List[Document]]:
        """Return key-value pairs given the text input to the chain.

        If None, return all memories
        """
        queries = inputs[self.queries_key]
        relevant_memories = [
            mem for query in queries for mem in self.fetch_memories(query)]
        relevant_memories_str = "\n".join(
            [f"{mem.page_content}" for mem in relevant_memories])
        return {self.answer_key: relevant_memories_str}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save the context of this model run to memory."""
        # TODO: fix the save memory key
        mem = outputs.get(self.add_key)
        if mem:
            self.add_memory(mem)

    def clear(self) -> None:
        """Clear memory contents."""
        # TODO
