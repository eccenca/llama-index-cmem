"""Testing cmem reader"""

from llama_index_cmem.readers.cmem import CMEMReader
from tests.conftest import GraphSetup
from tests.utils import needs_cmem

NUMBER_OF_DOCUMENTS = 10037


@needs_cmem
def test_cmem_reader(graph_setup: GraphSetup) -> None:
    """Testing cmem reader"""
    cmem_reader = CMEMReader()
    graph = graph_setup.graphs["combined"]["iri"]
    documents = cmem_reader.load_data(graph)
    assert len(documents) == NUMBER_OF_DOCUMENTS
