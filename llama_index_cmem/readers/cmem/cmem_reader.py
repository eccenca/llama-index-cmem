"""CMEM reader"""
import logging

from cmem.cmempy.queries import SparqlQuery, QUERY_STRING
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader

DEFAULT_QUERY = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?s ?sl ?pl ?ol
FROM <{graph}>
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

    def load_data(self, query: str, doc_id_binding: str, text_binding: list[str], metadata_binding: dict[str, str] = None, placeholder: dict = None) -> list[Document]:
        """Load data from SPARQL query response."""
        documents = []
        if query is None:
            logging.warning("No query provided.")
            return documents
        if doc_id_binding is None:
            logging.warning("No doc_id_binding provided.")
        if text_binding is None:
            logging.warning("No text_binding provided.")
        response = SparqlQuery(text=query).get_json_results(placeholder=placeholder)
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
                        if metadata_binding:
                            metadata = {}
                            for key in metadata_binding:
                                metadata[metadata_binding.get(key)] = binding[key]["value"]
                        documents.append(Document(doc_id=doc_id, text=text))
        return documents

    def load_default_data(self) -> list[Document]:
        """Loading all labels."""
        return self.load_data(DEFAULT_QUERY, DEFAULT_DOC_ID_BINDING, DEFAULT_TEXT_BINDING)

    def load_query_catalog_data(self) -> list[Document]:
        """Loading query catalog."""
        return self.load_data(QUERY_STRING, "query", ["label, description"], {"query": "identifier", "text": "sparql"})
