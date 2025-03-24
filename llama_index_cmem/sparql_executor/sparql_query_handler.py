"""SPARQL Query Handler"""

from typing import Any

from llama_index_cmem.sparql_executor.sparql_executor import SPARQLExecutor


class SPARQLQueryHandler:
    """Handles execution of SPARQL queries across different platforms."""

    def __init__(self, executor: SPARQLExecutor):
        self._executor = executor

    def run_sparql(self, query: str) ->dict[str, Any]:
        """Run sparql query"""
        try:
            return self._executor.run_query(query)
        except Exception as e:
            raise RuntimeError(f"Error executing SPARQL query: {e!s}") from e
