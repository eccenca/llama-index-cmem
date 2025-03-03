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

    Transforms SPARQL query results to llama-index documents.
    """

    def load_data(
        self,
        query: str,
        doc_id_binding: str,
        text_binding: list[str],
        metadata_binding: list[str] | None = None,
        placeholder: dict | None = None,
    ) -> list[Document]:
        """Load data from SPARQL query response."""
        documents = []
        if query is None:
            logging.warning("No query provided.")
            return documents
        if doc_id_binding is None:
            logging.warning("No doc_id_binding provided.")
        if text_binding is None:
            logging.warning("No text_binding provided.")
        response = SparqlQuery(query, query_type="SELECT").get_json_results(placeholder=placeholder)
        if response:
            results = response["results"]
            if results:
                bindings = results["bindings"]
                if bindings:
                    for binding in bindings:
                        doc_id = binding[doc_id_binding]["value"]
                        text = " ".join(
                            binding[key]["value"] for key in text_binding if key in binding
                        )
                        metadata = {}
                        if metadata_binding:
                            metadata = {
                                key: binding[key]["value"]
                                for key in metadata_binding
                                if key in binding
                            }
                        documents.append(Document(doc_id=doc_id, text=text, extra_info=metadata))
        return documents

    def load_default_data(self, placeholder: dict | None = None) -> list[Document]:
        """Load all labels."""
        return self.load_data(
            query=DEFAULT_QUERY,
            doc_id_binding=DEFAULT_DOC_ID_BINDING,
            text_binding=DEFAULT_TEXT_BINDING,
            placeholder=placeholder,
        )

    def load_query_catalog_data(self) -> list[Document]:
        """Load query catalog."""
        return self.load_data(
            query=QUERY_STRING,
            doc_id_binding="query",
            text_binding=["label", "description"],
            metadata_binding=["query", "label", "description"],
        )
