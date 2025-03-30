"""SPARQL Reader"""

import logging

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader

from llama_index_cmem.executor.sparql_executor import SPARQLExecutor
from llama_index_cmem.executor.sparql_query_handler import SPARQLQueryHandler

GRAPH_LABELS_QUERY = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?s ?sl ?pl ?ol
FROM <{{graph}}>
WHERE {{
  ?s ?p ?o .
  ?s rdfs:label ?sl .
  ?p rdfs:label ?pl .
  {{
    OPTIONAL {{
      OPTIONAL {{ ?o rdfs:label ?ol_ . }}
      BIND(IF(!ISIRI(?o), ?o, ?ol_) AS ?ol)
    }}
  }}
}}
"""


class SPARQLReader(BaseReader):
    """SPARQL Reader for transforming SPARQL query results into LlamaIndex Documents."""

    def __init__(self, executor: SPARQLExecutor):
        """Initialize the reader with a SPARQLExecutor."""
        self.executor = executor
        self._handler = SPARQLQueryHandler(executor=executor)

    def load_data(
        self,
        query: str,
        doc_id_binding: str,
        text_binding: list[str],
        metadata_binding: list[str] | None = None,
    ) -> list[Document]:
        """Execute a SPARQL query and return the results as LlamaIndex Documents.

        :param query: The SPARQL query to execute.
        :param doc_id_binding: The binding variable used for document IDs.
        :param text_binding: The list of binding variables that will form the document text.
        :param metadata_binding: Optional list of metadata bindings to include in the document.
        :return: A list of LlamaIndex Document objects.
        """
        if not query:
            logging.warning("No query provided.")
            return []

        if not doc_id_binding or not text_binding:
            logging.warning("Both doc_id_binding and text_binding must be provided.")
            return []

        if metadata_binding is None:
            metadata_binding = []

        # Fetch the SPARQL query response
        try:
            response = self._handler.run_sparql(query)
        except Exception:
            logging.exception("Error executing SPARQL query")
            return []

        # If the response is empty or invalid, return an empty list
        if not response or not response.get("results", {}).get("bindings"):
            logging.warning("No results found in the SPARQL response.")
            return []

        # Process the results and create Document objects
        return self._parse_bindings(
            response["results"]["bindings"], doc_id_binding, text_binding, metadata_binding
        )

    @staticmethod
    def _parse_bindings(
        bindings: list[dict],
        doc_id_binding: str,
        text_binding: list[str],
        metadata_binding: list[str],
    ) -> list[Document]:
        """Parse the SPARQL bindings and convert them to Document objects.

        :param bindings: The SPARQL result bindings.
        :param doc_id_binding: The binding to use for the document ID.
        :param text_binding: The list of bindings to extract for document text.
        :param metadata_binding: The list of metadata bindings.
        :return: A list of LlamaIndex Document objects.
        """
        documents = []
        for binding in bindings:
            doc_id = binding.get(doc_id_binding, {}).get("value", "")
            text = " ".join(
                binding.get(key, {}).get("value", "") for key in text_binding if key in binding
            )
            metadata = {
                key: binding.get(key, {}).get("value", "")
                for key in (metadata_binding or [])
                if key in binding
            }

            if doc_id and text:
                documents.append(Document(doc_id=doc_id, text=text, extra_info=metadata))
        return documents

    def load_graph_triples_with_labels(self, graph: str) -> list[Document]:
        """Load Fetch subjects, predicates, and objects with their labels from the graph.

        :param graph: The graph URI for the SPARQL query.
        :return: A list of LlamaIndex Documents.
        """
        query = GRAPH_LABELS_QUERY.replace("{{graph}}", graph)
        return self.load_data(
            query=query,
            doc_id_binding="s",  # Assuming 's' is the subject in the SPARQL result
            text_binding=["sl", "pl", "ol"],  # Assuming we want these fields as text
            metadata_binding=["s", "pl", "ol"],
            # Metadata like subject, predicate, object labels
        )
