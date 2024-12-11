from typing import Any

from cmem.cmempy.queries import SparqlQuery
from llama_index.core.graph_stores.types import GraphStore


def is_empty_result(response: Any):
    return len(response.get("results").get("bindings")) < 1


class CMEMGraphStore(GraphStore):
    def query(self, query: str, param_map: dict[str, Any] | None = None) -> Any:
        print("SPARQL Query: " + query)
        response = SparqlQuery(query).get_json_results()
        print("CMEM Response: " + str(response))
        return response
