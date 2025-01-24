"""Testing cmem reader"""

from llama_index_cmem.readers.cmem import CMEMReader

NUMBER_OF_DOCUMENTS = 20043


def test_cmem_reader() -> None:
    """Testing cmem reader"""
    cmem_reader = CMEMReader()
    graph = "<http://ld.company.org/prod-inst/>"
    documents = cmem_reader.load_data(graph)
    assert len(documents) == NUMBER_OF_DOCUMENTS
