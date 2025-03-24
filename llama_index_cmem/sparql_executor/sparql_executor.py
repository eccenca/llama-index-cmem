"""SPARQL Executor Module"""

from abc import ABC, abstractmethod
from typing import Any


class SPARQLExecutor(ABC):
    """Abstract base class for executing SPARQL queries on various platforms."""

    @abstractmethod
    def run_query(self, query: str) -> list[dict[str, Any]]:
        """Execute a SPARQL query and returns the results.

        Args:
            query (str): The SPARQL query string to be executed.

        Returns:
            list[dict[str, Any]]: A list of dictionaries representing query results,
                                  where each dictionary corresponds to a result row.

        """
