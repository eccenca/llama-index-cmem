"""Testing cmem retriever"""

from collections.abc import Generator
from shutil import rmtree
from tempfile import mkdtemp
from typing import TYPE_CHECKING

import pytest
from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI

from llama_index_cmem.retrievers.cmem.cmem_retriever import CMEMRetriever
from tests.cmemc_command_utils import run, run_without_assertion
from tests.utils import needs_cmem, needs_openai

if TYPE_CHECKING:
    from llama_index_cmem.utils.cmem_query import CMEMQuery


@pytest.fixture
def chat_engine() -> CondensePlusContextChatEngine:
    """Provide test chat engine for CMEM"""
    model = "gpt-4o-mini"
    llm = OpenAI(model=model)
    Settings.llm = llm
    ontology_graph = "http://ld.company.org/prod-vocab/"
    context_graph = "http://ld.company.org/prod-inst/"
    get_response_synthesizer()
    retriever = CMEMRetriever(ontology_graph=ontology_graph, context_graph=context_graph, llm=llm)
    memory = ChatMemoryBuffer.from_defaults()
    return CondensePlusContextChatEngine(retriever=retriever, llm=llm, memory=memory)


graphs = {
    "vocab": {
        "location": "https://download.eccenca.com/testing-assets/products-demo-project/prod-vocab.ttl",
        "iri": "http://ld.company.org/prod-vocab/",
    },
    "dataset": {
        "location": "https://download.eccenca.com/testing-assets/products-demo-project/prod-inst.ttl",
        "iri": "http://ld.company.org/prod-inst/",
    },
}


@pytest.fixture
def graph_setup() -> Generator[None, None, None]:
    """Graph setup fixture"""
    # make backup and delete all graphs
    backup_directory = mkdtemp(prefix="cmemc-graphs-backup")
    for _ in graphs.values():
        run_without_assertion(["graph", "export", "--output-dir", backup_directory, _["iri"]])
        run_without_assertion(["graph", "delete", _["iri"]])
        run(["graph", "import", _["location"], _["iri"]])

    yield None
    # remove test graphs
    for _ in graphs.values():
        run(["graph", "delete", _["iri"]])

    # import backup graphs and compare triple counts
    run(["graph", "import", backup_directory])
    rmtree(backup_directory)


@needs_openai
@needs_cmem
@pytest.mark.parametrize("limit", [10, 20])
@pytest.mark.usefixtures("graph_setup")
def test_cmem_retriever(chat_engine: CondensePlusContextChatEngine, limit: int) -> None:
    """Test cmem retriever"""
    prompt = f"List all hardware with price. Limit the results to {limit} items."
    response = chat_engine.chat(prompt)
    assert len(response.source_nodes) == 1
    node = response.source_nodes[0]
    cmem_query: CMEMQuery = node.metadata["cmem_query"]
    assert cmem_query.get_last_sparql()
    assert len(node.metadata["cmem_response"]["results"]["bindings"]) == limit
