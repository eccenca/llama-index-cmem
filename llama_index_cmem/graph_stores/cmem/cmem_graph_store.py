"""CMEM graph store"""

import logging
from typing import Any

from cmem.cmempy.queries import SparqlQuery
from llama_index.core.graph_stores.types import GraphStore

logger = logging.getLogger(__name__)


def is_empty_result(response: dict) -> bool:
    """Check if a cmem response is empty."""
    return len(response.get("results").get("bindings")) < 1


class CMEMGraphStore(GraphStore):
    """CMEM Graph Store implementation.

    Runs a SPARQL query against a cmem instance.
    """

    def query(self, query: str, param_map: dict[str, Any] | None = None) -> dict:
        """Query CMEM graph store"""
        placeholder = None
        if param_map:
            placeholder = param_map["placeholder"]
        logger.info(f"SPARQL Query: {0}".format(query))
        response = SparqlQuery(query, placeholder=placeholder).get_json_results()
        logger.info(f"CMEM Response: {response!s}")
        return response
