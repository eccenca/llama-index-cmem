"""CMEM reader"""

from cmem.cmempy.queries import SparqlQuery
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader

ALL_LABELS_QUERY = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?s ?sl ?pl ?ol
FROM <http://ld.company.org/prod-inst/>
WHERE {
  ?s ?p ?o .
  ?s rdfs:label ?sl .
  ?p rdfs:label ?pl .
  { OPTIONAL
    { ?o rdfs:label ?ol_ . }
    BIND(IF(!ISIRI(?o), ?o, ?ol_) AS ?ol)
  }
}
"""


class CMEMReader(BaseReader):
    """CMEM reader implementation.

    Transforms SPARQL query results to llama-index documents.
    """

    def load_data(self) -> list[Document]:
        """Load data from SPARQL query response."""
        response = SparqlQuery(ALL_LABELS_QUERY).get_json_results()
        documents = []
        if response:
            results = response["results"]
            if results:
                bindings = results["bindings"]
                if bindings:
                    for binding in bindings:
                        doc_id = binding["s"]["value"]
                        text = " ".join(
                            binding[key]["value"] for key in ["sl", "pl", "ol"] if key in binding
                        )
                        documents.append(Document(doc_id=doc_id, text=text))

        return documents
