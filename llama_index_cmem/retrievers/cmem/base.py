"""llama_index retrievers cmem base"""

import json
from abc import ABC, abstractmethod

from llama_index.core import QueryBundle
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode


class CMEMBaseRetriever(BaseRetriever, ABC):
    """Abstract Base Retriever for CMEM"""

    def __init__(self, score: float = 1.0):
        super().__init__()
        self.score = score  # Default score for retrieved nodes

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve CMEM results, converts them, and applies post-processing."""
        raw_results, metadata = self._retrieve_cmem_results(query_bundle)
        return self._convert_results(raw_results, metadata)

    @abstractmethod
    def _retrieve_cmem_results(self, query_bundle: QueryBundle) -> tuple[dict, dict]:
        """Must be implemented by subclasses to fetch results from CMEM."""

    def _convert_results(self, results: dict, metadata: dict | None = None) -> list[NodeWithScore]:
        """Convert a results dictionary to a list containing a single NodeWithScore object."""
        if not results:
            return []
        metadata = metadata or {}
        node = TextNode(text=json.dumps(results), metadata=metadata)
        return [NodeWithScore(node=node, score=self.score)]
