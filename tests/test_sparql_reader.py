"""Testing cmem reader"""

from cmem.cmempy.queries import QUERY_STRING

from llama_index_cmem.executor.cmem_sparql_executor import CMEMSPARQLExecutor
from llama_index_cmem.readers.sparql_reader import SPARQLReader
from tests.conftest import GraphSetup
from tests.utils import needs_cmem

NUMBER_OF_DOCUMENTS = 10037

NUMBER_OF_QUERIES = 5


@needs_cmem
def test_sparql_reader(graph_setup: GraphSetup) -> None:
    """Testing cmem reader"""
    sparql_reader = SPARQLReader(executor=CMEMSPARQLExecutor())
    graph = graph_setup.graphs["combined"]["iri"]
    documents = sparql_reader.load_graph_triples_with_labels(graph=graph)
    assert len(documents) == NUMBER_OF_DOCUMENTS


def test_sparql_reader_query_catalog(graph_setup: GraphSetup) -> None:
    """Testing cmem reader query catalog"""
    assert graph_setup.graphs["query_catalog"]
    sparql_reader = SPARQLReader(executor=CMEMSPARQLExecutor())
    documents = sparql_reader.load_data(
        query=QUERY_STRING,
        doc_id_binding="query",
        text_binding=["label", "description"],
        metadata_binding=["query", "label", "description"],
    )
    assert len(documents) == NUMBER_OF_QUERIES
