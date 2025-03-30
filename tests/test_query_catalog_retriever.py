"""Testing catalog retriever"""

import pytest
from llama_index.core import QueryBundle, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from llama_index_cmem.executor.cmem_sparql_executor import CMEMSPARQLExecutor
from llama_index_cmem.retrievers.cmem_query_catalog_retriever import CMEMQueryCatalogRetriever
from llama_index_cmem.retrievers.sparql_retriever import SPARQLRetriever

CATALOG_RETRIEVER_PROPERTIES_BINDINGS = 33
CATALOG_RETRIEVER_CLASSES_BINDINGS = 11


@pytest.fixture(scope="module")
def setup_llm() -> None:
    """Set llm and embedding"""
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    model = "gpt-4o-mini"
    llm = OpenAI(model=model)
    Settings.llm = llm


@pytest.mark.usefixtures("graph_setup", "setup_llm")
def test_catalog_retriever_services() -> None:
    """Test catalog retriever without placeholders and services"""
    catalog_retriever = CMEMQueryCatalogRetriever()
    retrieved_nodes = catalog_retriever.retrieve(QueryBundle(query_str=":all-services"))
    assert len(retrieved_nodes) == 1
    query = retrieved_nodes[0].metadata["query"]
    assert query == "https://ns.eccenca.com/data/queries/all-services"


@pytest.mark.usefixtures("graph_setup", "setup_llm")
def test_catalog_retriever_hardware() -> None:
    """Test catalog retriever without placeholders and hardware"""
    catalog_retriever = CMEMQueryCatalogRetriever()
    retrieved_nodes = catalog_retriever.retrieve(QueryBundle(query_str=":all-hardware"))
    assert len(retrieved_nodes) == 1
    query = retrieved_nodes[0].metadata["query"]
    assert query == "https://ns.eccenca.com/data/queries/all-hardware"


@pytest.mark.usefixtures("graph_setup", "setup_llm")
def test_catalog_retriever_with_placeholder_classes() -> None:
    """Test catalog retriever with placeholders and classes"""
    catalog_retriever = CMEMQueryCatalogRetriever()
    retrieved_nodes = catalog_retriever.retrieve(
        QueryBundle(query_str=":list-classes for graph http://ld.company.org/prod-inst/")
    )
    assert len(retrieved_nodes) == 1
    identifier = retrieved_nodes[0].metadata["query"]
    assert identifier == "https://ns.eccenca.com/data/queries/list-classes"
    sparql_retriever = SPARQLRetriever(CMEMSPARQLExecutor(), return_raw=False)
    retrieved_nodes = sparql_retriever.retrieve(
        QueryBundle(query_str=retrieved_nodes[0].metadata["text"])
    )
    assert len(retrieved_nodes) == CATALOG_RETRIEVER_CLASSES_BINDINGS


@pytest.mark.usefixtures("graph_setup", "setup_llm")
def test_catalog_retriever_with_placeholder_properties() -> None:
    """Test catalog retriever with placeholders and properties"""
    catalog_retriever = CMEMQueryCatalogRetriever()
    retrieved_nodes = catalog_retriever.retrieve(
        QueryBundle(query_str=":list-properties for graph http://ld.company.org/prod-inst/")
    )
    query = retrieved_nodes[0].metadata["query"]
    assert query == "https://ns.eccenca.com/data/queries/list-properties"
    sparql_retriever = SPARQLRetriever(CMEMSPARQLExecutor(), return_raw=False)
    retrieved_nodes = sparql_retriever.retrieve(
        QueryBundle(query_str=retrieved_nodes[0].metadata["text"])
    )
    assert len(retrieved_nodes) == CATALOG_RETRIEVER_PROPERTIES_BINDINGS


@pytest.mark.usefixtures("graph_setup", "setup_llm")
def test_catalog_retriever_vector_retrieve_services() -> None:
    """Test catalog retriever using vectorstore retrieve"""
    catalog_retriever = CMEMQueryCatalogRetriever()
    retrieved_nodes = catalog_retriever.retrieve(
        QueryBundle(query_str="Show me all service from product demo.")
    )
    assert len(retrieved_nodes) == 1
    query = retrieved_nodes[0].metadata["query"]
    assert query == "https://ns.eccenca.com/data/queries/all-services"
