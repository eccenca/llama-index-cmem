"""CMEM reader"""

import logging

from cmem.cmempy.queries import QUERY_STRING, SparqlQuery
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader

DEFAULT_QUERY = """
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

DEFAULT_DOC_ID_BINDING = "s"

DEFAULT_TEXT_BINDING = ["sl", "pl", "ol"]


class CMEMReader(BaseReader):
    """CMEM reader implementation.

    Transforms SPARQL query results into llama-index documents.
    """

    def load_data(
        self,
        query: str,
        doc_id_binding: str,
        text_binding: list[str],
        metadata_binding: list[str] | None = None,
    ) -> list[Document]:
        """Load data from SPARQL query response."""
        if not query:
            logging.warning("No query provided.")
            return []

        if not doc_id_binding:
            logging.warning("No doc_id_binding provided.")

        if not text_binding:
            logging.warning("No text_binding provided.")

        if not metadata_binding:
            metadata_binding = []

        # Fetch the SPARQL query response
        response = SparqlQuery(query, query_type="SELECT").get_json_results()

        # If the response is empty or invalid, return an empty list
        if not response or not response.get("results", {}).get("bindings"):
            logging.warning("No results found in the SPARQL response.")
            return []

        # Process the results and create Document objects
        return self._parse_bindings(
            response["results"]["bindings"], doc_id_binding, text_binding, metadata_binding
        )

    def load_graph_data(self, graph: str) -> list[Document]:
        """Load all labels with default parameters."""
        return self.load_data(
            query=SparqlQuery(DEFAULT_QUERY).get_filled_text(placeholder={"graph": graph}),
            doc_id_binding=DEFAULT_DOC_ID_BINDING,
            text_binding=DEFAULT_TEXT_BINDING,
        )

    def load_query_catalog_data(self) -> list[Document]:
        """Load query catalog with specific bindings."""
        return self.load_data(
            query=QUERY_STRING,
            doc_id_binding="query",
            text_binding=["label", "description"],
            metadata_binding=["query", "label", "description"],
        )

    @staticmethod
    def _parse_bindings(
        bindings: list[dict],
        doc_id_binding: str,
        text_binding: list[str],
        metadata_binding: list[str],
    ) -> list[Document]:
        """Parse the SPARQL bindings and convert them to Document objects."""
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
