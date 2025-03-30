"""test sparql retriever"""

import pytest
from llama_index.core import QueryBundle, Settings
from llama_index.llms.openai import OpenAI

from llama_index_cmem.executor.cmem_sparql_executor import CMEMSPARQLExecutor
from llama_index_cmem.retrievers.sparql_retriever import NLSPARQLRetriever, SPARQLRetriever
from llama_index_cmem.utils.cmem_query_builder import download_ontology
from tests.utils import needs_openai

LIMIT = 10
ONTOLOGY_GRAPH = "http://ld.company.org/prod-vocab/"
CONTEXT_GRAPH = "http://ld.company.org/prod-inst/"
SAMPLE_QUERY = f"""PREFIX  rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX  rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT  *
WHERE
  {{ ?sub  rdfs:label  ?obj }}
LIMIT   {LIMIT}
"""


@pytest.mark.usefixtures("graph_setup")
def test_sparql_retriever() -> None:
    """Test sparql retriever"""
    retriever = SPARQLRetriever(CMEMSPARQLExecutor())
    assert len(retriever.retrieve(QueryBundle(query_str=SAMPLE_QUERY))) == 1

    retriever = SPARQLRetriever(CMEMSPARQLExecutor(), return_raw=False)
    assert len(retriever.retrieve(QueryBundle(query_str=SAMPLE_QUERY))) == LIMIT


@needs_openai
@pytest.mark.usefixtures("graph_setup")
def test_nl_sparql_retriever() -> None:
    """Test nl sparql retriever"""
    model = "gpt-4o-mini"
    llm = OpenAI(model=model)
    Settings.llm = llm

    retriever = NLSPARQLRetriever(
        executor=CMEMSPARQLExecutor(), ontology_triples=download_ontology(ONTOLOGY_GRAPH), llm=llm
    )
    response = retriever.retrieve(
        QueryBundle(
            query_str=f"List all hardware with price. Limit the results to {LIMIT}"
            f" items from {CONTEXT_GRAPH}"
        )
    )
    assert len(response) == 1

    retriever = NLSPARQLRetriever(
        executor=CMEMSPARQLExecutor(),
        llm=llm,
        ontology_triples=download_ontology(ONTOLOGY_GRAPH),
        return_raw=False,
    )
    response = retriever.retrieve(
        QueryBundle(
            query_str=f"List all hardware with price. Limit the results to {LIMIT}"
            f" items from {CONTEXT_GRAPH}"
        )
    )
    assert len(response) == LIMIT
