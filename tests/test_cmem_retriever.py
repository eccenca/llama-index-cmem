"""Testing cmem retriever"""

from typing import TYPE_CHECKING

import pytest
from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI

from llama_index_cmem.retrievers.cmem.cmem_retriever import CMEMRetriever
from tests.utils import needs_openai

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


@needs_openai
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


@pytest.mark.usefixtures("graph_setup")
def test_catalog_retriever() -> None:
    """Test catalog retriever"""
