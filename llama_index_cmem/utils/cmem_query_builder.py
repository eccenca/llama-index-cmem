"""CMEM query builder"""

from cmem.cmempy.dp.proxy.graph import get
from llama_index.core import PromptTemplate
from llama_index.core.llms import LLM
from llama_index.core.prompts import PromptType

from llama_index_cmem.utils.cmem_query import CMEMQuery, as_sparql

DEFAULT_SPARQL_PROMPT_TEMPLATE = """
You are an expert for generating SPARQL queries to answer a question.
The original question is given below.
The RDF ontology in turtle format is given below.
Generate a valid SPARQL query considering the given ontology
to answer the question using this graph '{context_graph}'.
Original question: {query_str}
RDF ontology: {ontology_str}
Response:
"""

DEFAULT_SPARQL_PROMPT = PromptTemplate(
    DEFAULT_SPARQL_PROMPT_TEMPLATE,
    prompt_type=PromptType.TEXT_TO_GRAPH_QUERY,
)

DEFAULT_SPARQL_REFINE_PROMPT_TEMPLATE = """
You are an expert for refining SPARQL queries to answer a user question.

The user question is given below.
The RDF ontology in turtle format is given below.
The original SPARQL query is given below.

The original SPARQL query did not work as expected.
Refine the given SPARQL query considering the given ontology
to answer the user question using this graph '{context_graph}'.

User question: {query_str}
RDF ontology: {ontology_str}
Original SPARQL query: {sparql_str}
Response:
"""

DEFAULT_SPARQL_REFINE_PROMPT = PromptTemplate(
    DEFAULT_SPARQL_REFINE_PROMPT_TEMPLATE, prompt_type=PromptType.REFINE
)


def download_ontology(ontology_graph: str) -> str:
    """Download an ontology as text/turtle"""
    graph = get(ontology_graph, owl_imports_resolution=True, accept="text/turtle")
    return graph.content


class LLMQueryBuilder:
    """LLM query builder.

    This query builder generates SPARQL queries based on a natural language and a given ontology.

    """

    def __init__(self, ontology_graph: str, context_graph: str, llm: LLM):
        self.ontology_graph = ontology_graph
        self.context_graph = context_graph
        self.llm = llm
        self.ontology_str = download_ontology(self.ontology_graph)

    def generate_sparql(self, question: str) -> CMEMQuery:
        """Generate SPARQL query"""
        predict = self.llm.predict(
            DEFAULT_SPARQL_PROMPT,
            query_str=question,
            ontology_str=self.ontology_str,
            context_graph=self.context_graph,
        )
        cmem_query = CMEMQuery(question)
        cmem_query.set_sparql_prediction(predict)
        return cmem_query

    def refine_sparql(self, question: str, cmem_query: CMEMQuery) -> CMEMQuery:
        """Refine SPARQL query"""
        predict = self.llm.predict(
            DEFAULT_SPARQL_REFINE_PROMPT,
            query_str=question,
            ontology_str=self.ontology_str,
            context_graph=self.context_graph,
            sparql_str=as_sparql(cmem_query.get_sparql()),
        )
        cmem_query.set_refined_prediction(predict)
        return cmem_query
