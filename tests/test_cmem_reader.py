"""Testing cmem reader"""

from llama_index_cmem.readers.cmem import CMEMReader
from tests.conftest import GraphSetup
from tests.utils import needs_cmem

NUMBER_OF_DOCUMENTS = 10037

NUMBER_OF_QUERIES = 5


@needs_cmem
def test_cmem_reader(graph_setup: GraphSetup) -> None:
    """Testing cmem reader"""
    cmem_reader = CMEMReader()
    graph = graph_setup.graphs["combined"]["iri"]
    documents = cmem_reader.load_default_data(placeholder={"graph": graph})
    assert len(documents) == NUMBER_OF_DOCUMENTS


def test_cmem_reader_query_catalog(graph_setup: GraphSetup) -> None:
    """Testing cmem reader query catalog"""
    assert graph_setup.graphs["query_catalog"]
    cmem_reader = CMEMReader()
    documents = cmem_reader.load_query_catalog_data()
    assert len(documents) == NUMBER_OF_QUERIES
