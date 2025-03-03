"""Testing catalog retriever"""

import json
from typing import Any

import pytest
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore

from llama_index_cmem.retrievers.cmem.catlog import CatalogRetriever

CATALOG_RETRIEVER_PROPERTIES_BINDINGS = 33

CATALOG_RETRIEVER_CLASSES_BINDINGS = 11

CATALOG_RETRIEVER_HARDWARE_BINDINGS = 1000

CATALOG_RETRIEVER_SERVICES_BINDINGS = 9


def extract_bindings(nodes_with_score: list[NodeWithScore]) -> list[Any]:
    """Extract bindings from nodes"""
    text: dict[str, Any] = json.loads(nodes_with_score[0].text)
    results = text.get("results")
    if not isinstance(results, dict):
        return []
    bindings = results.get("bindings")
    if not isinstance(bindings, list):
        return []
    return [item for item in bindings if isinstance(item, dict)]


def extract_metadata(nodes_with_score: list[NodeWithScore]) -> dict[str, Any]:
    """Extract metadata from nodes"""
    return nodes_with_score[0].metadata


@pytest.mark.usefixtures("graph_setup")
def test_catalog_retriever_unknown_query() -> None:
    """Test catalog retriever with an unknown query"""
    catalog_retriever = CatalogRetriever(identifier=":unknown-query")
    retrieved_nodes = catalog_retriever.retrieve(QueryBundle(query_str="n/a"))
    assert len(retrieved_nodes) == 0


@pytest.mark.usefixtures("graph_setup")
def test_catalog_retriever_services() -> None:
    """Test catalog retriever without placeholders and services"""
    catalog_retriever = CatalogRetriever(identifier=":all-services")
    retrieved_nodes = catalog_retriever.retrieve(QueryBundle(query_str="n/a"))
    assert len(retrieved_nodes) == 1
    bindings = extract_bindings(retrieved_nodes)
    assert len(bindings) == CATALOG_RETRIEVER_SERVICES_BINDINGS


@pytest.mark.usefixtures("graph_setup")
def test_catalog_retriever_hardware() -> None:
    """Test catalog retriever without placeholders and hardware"""
    catalog_retriever = CatalogRetriever(identifier=":all-hardware")
    retrieved_nodes = catalog_retriever.retrieve(QueryBundle(query_str="n/a"))
    assert len(retrieved_nodes) == 1
    bindings = extract_bindings(retrieved_nodes)
    assert len(bindings) == CATALOG_RETRIEVER_HARDWARE_BINDINGS


@pytest.mark.usefixtures("graph_setup")
def test_catalog_retriever_with_placeholder_classes() -> None:
    """Test catalog retriever with placeholders and classes"""
    placeholder = {"graph": "http://ld.company.org/prod-inst/"}
    catalog_retriever = CatalogRetriever(identifier=":list-classes", placeholder=placeholder)
    retrieved_nodes = catalog_retriever.retrieve(QueryBundle(query_str="n/a"))
    assert len(retrieved_nodes) == 1
    bindings = extract_bindings(retrieved_nodes)
    assert len(bindings) == CATALOG_RETRIEVER_CLASSES_BINDINGS


@pytest.mark.usefixtures("graph_setup")
def test_catalog_retriever_with_placeholder_properties() -> None:
    """Test catalog retriever with placeholders and properties"""
    placeholder = {"graph": "http://ld.company.org/prod-inst/"}
    catalog_retriever = CatalogRetriever(identifier=":list-properties", placeholder=placeholder)
    retrieved_nodes = catalog_retriever.retrieve(QueryBundle(query_str="n/a"))
    assert len(retrieved_nodes) == 1
    bindings = extract_bindings(retrieved_nodes)
    assert len(bindings) == CATALOG_RETRIEVER_PROPERTIES_BINDINGS


@pytest.mark.usefixtures("graph_setup")
def test_catalog_retriever_auto_select_services() -> None:
    """Test catalog retriever using auto select without placeholders and services"""
    catalog_retriever = CatalogRetriever()
    retrieved_nodes = catalog_retriever.retrieve(
        QueryBundle(query_str="Show me all services from product demo.")
    )
    assert len(retrieved_nodes) == 1
    metadata = extract_metadata(retrieved_nodes)
    assert metadata["cmem"]["identifier"] == ":all-services"
    bindings = extract_bindings(retrieved_nodes)
    assert len(bindings) == CATALOG_RETRIEVER_SERVICES_BINDINGS


@pytest.mark.usefixtures("graph_setup")
def test_catalog_retriever_auto_select_with_placeholder_classes() -> None:
    """Test catalog retriever with auto select with placeholders and classes"""
    catalog_retriever = CatalogRetriever()
    retrieved_nodes = catalog_retriever.retrieve(
        QueryBundle(
            query_str="Show me used classes in this graph: 'http://ld.company.org/prod-inst/'"
        )
    )
    assert len(retrieved_nodes) == 1
    metadata = extract_metadata(retrieved_nodes)
    assert metadata["cmem"]["identifier"] == ":list-classes"
    assert (
        metadata["cmem"]["placeholder"]["graph"] == "http://ld.company.org/prod-inst/"
    )
    bindings = extract_bindings(retrieved_nodes)
    assert len(bindings) == CATALOG_RETRIEVER_CLASSES_BINDINGS

@pytest.mark.usefixtures("graph_setup")
def test_catalog_retriever_auto_select_services() -> None:
    """Test catalog retriever using auto select without placeholders and services"""
    catalog_retriever = CatalogRetriever()
    retrieved_nodes = catalog_retriever.retrieve(
        QueryBundle(query_str="xxx")
    )
    assert len(retrieved_nodes) == 0
    metadata = extract_metadata(retrieved_nodes)
    assert metadata["cmem"]["identifier"] == ":all-services"
    bindings = extract_bindings(retrieved_nodes)
    assert len(bindings) == CATALOG_RETRIEVER_SERVICES_BINDINGS
