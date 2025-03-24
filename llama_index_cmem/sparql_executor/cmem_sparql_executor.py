"""CMEM SPARQL Executor."""

from typing import Any

from cmem.cmempy.queries import SparqlQuery

from llama_index_cmem.sparql_executor.sparql_executor import SPARQLExecutor


class CMEMSPARQLExecutor(SPARQLExecutor):
    """Executes SPARQL queries on the CMEM (Corporate Memory) platform."""

    def __init__(self):
        super().__init__()
        self._sparqlquery = SparqlQuery(text="", query_type="SELECT")

    def run_query(self, query: str) -> dict[str, Any]:
        """Run sparql query"""
        self._sparqlquery.text = query
        return self._sparqlquery.get_json_results()  # type: ignore[no-any-return]
