"""CMEM query builder 2"""

from cmem.cmempy.dp.proxy.graph import get
from llama_index.core import PromptTemplate, Settings
from llama_index.core.llms import LLM
from llama_index.core.prompts import PromptType

from llama_index_cmem.utils.cmem_query import CMEMQuery, format_sparql_list

DEFAULT_SPARQL_PROMPT_TEMPLATE = """
You are an expert for generating SPARQL queries.

Follow this rules when generating SPARQL queries:
A SPARQL query needs to answer a specific user question.
A SPARQL query needs to follow a given RDF ontology.
A SPARQL query should be explained.

The user question is given below.
The RDF ontology in turtle format is given below.

Generate a valid SPARQL query considering the given ontology
to answer the user question using this graph '{context_graph}'.

User question: {query_str}
RDF ontology: {ontology_str}
Response:
"""

DEFAULT_SPARQL_PROMPT = PromptTemplate(
    DEFAULT_SPARQL_PROMPT_TEMPLATE,
    prompt_type=PromptType.TEXT_TO_GRAPH_QUERY,
)

DEFAULT_SPARQL_REFINE_PROMPT_TEMPLATE = """
You are an expert for generating and refining SPARQL queries.
The original/previous SPARQL query did not work as expected, so we need to refine it.

Follow this rules when generating SPARQL queries:
A SPARQL query needs to answer a specific user question.
A SPARQL query needs to follow a given RDF ontology.
A SPARQL query should be explained.
A refined SPARQL query needs to differ from any previous SPARQL query.

The user question is given below.
The RDF ontology in turtle format is given below.
All previous SPARQL queries are given below.

Generate a valid SPARQL query considering the given ontology
to answer the question using this graph '{context_graph}'.

User question: {query_str}
RDF ontology: {ontology_str}
All previous SPARQL queries: {sparql_str}
Response:
"""

DEFAULT_SPARQL_REFINE_PROMPT = PromptTemplate(
    DEFAULT_SPARQL_REFINE_PROMPT_TEMPLATE, prompt_type=PromptType.REFINE
)


def download_ontology(ontology_graph: str) -> str:
    """Download an ontology as text/turtle"""
    graph = get(ontology_graph, owl_imports_resolution=True, accept="text/turtle")
    return str(graph.content)


class CMEMQueryBuilder:
    """CMEM query builder.

    This query builder generates SPARQL queries based on a natural language and a given ontology.

    """

    def __init__(self, ontology_graph: str, context_graph: str, llm: LLM):
        self.ontology_graph = ontology_graph
        self.context_graph = context_graph
        self.llm = llm or Settings.llm
        self.ontology_str = download_ontology(self.ontology_graph)

    def generate_sparql(
        self, question: str, prompt: PromptTemplate = DEFAULT_SPARQL_PROMPT
    ) -> CMEMQuery:
        """Generate SPARQL query"""
        predict = self.llm.predict(
            prompt=prompt,
            query_str=question,
            ontology_str=self.ontology_str,
            context_graph=self.context_graph,
        )
        cmem_query = CMEMQuery(question)
        cmem_query.add(predict)
        return cmem_query

    def refine_sparql(
        self,
        question: str,
        cmem_query: CMEMQuery,
        prompt: PromptTemplate = DEFAULT_SPARQL_REFINE_PROMPT,
    ) -> CMEMQuery:
        """Refine SPARQL query"""
        sparql_str = format_sparql_list(cmem_query.get_sparql_list())
        predict = self.llm.predict(
            prompt=prompt,
            query_str=question,
            ontology_str=self.ontology_str,
            context_graph=self.context_graph,
            sparql_str=sparql_str,
        )
        cmem_query.add(predict)
        return cmem_query
