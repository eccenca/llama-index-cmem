"""SPARQL Executor Module"""

from abc import ABC, abstractmethod
from typing import Any


class SPARQLExecutor(ABC):
    """Abstract base class for executing SPARQL queries on various platforms."""

    @abstractmethod
    def run_query(self, query: str) -> dict[str, Any]:
        """Execute a SPARQL query and returns the results."""
