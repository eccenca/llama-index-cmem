"""Prompts module"""

from llama_index.core import PromptTemplate
from llama_index.core.prompts import PromptType

DEFAULT_TEXT_TO_SPARQL_PROMPT_TEMPLATE = """
You are an expert in SPARQL and semantic web technologies. Below is an ontology schema
 in RDF triples. Generate an optimized SPARQL query that answers the users question
 using this ontology.


### **Generate a SPARQL Query that Retrieves Relevant Information**
Ensure the query:
- Uses **appropriate ontology classes and properties**.
- Is **logically sound** and **efficient**.
- Returns **accurate** and **relevant results**.

You are required to use the following format, each taking one line:\n\n
Question: Question here\n
SPARQLQuery: SQL Query to run with prefix\n
SPARQLResult: Result of the SPARQLQuery\
Answer: Final answer here\n\n
Only Ontology (RDF Triples) listed below.\n
{ontology_triples}\n\n
Question: {question}\n
SPARQLQuery: \
"""

DEFAULT_TEXT_TO_SPARQL_PROMPT = PromptTemplate(
    DEFAULT_TEXT_TO_SPARQL_PROMPT_TEMPLATE,
    prompt_type=PromptType.TEXT_TO_GRAPH_QUERY,
)

# LLM prompt to select the best query
DEFAULT_SPARQL_QUERY_SELECTION_PROMPT_TEMPLATE = """
You are an expert in selecting SPARQL queries.
Given the user's request, choose the most relevant query.

User Request: {user_query}

Available Queries:
{query_options}

Respond with only the query number that best matches the user's intent.
"""
DEFAULT_SPARQL_QUERY_SELECTION_PROMPT = PromptTemplate(
    template=DEFAULT_SPARQL_QUERY_SELECTION_PROMPT_TEMPLATE,
)


DEFAULT_PLACEHOLDER_QUERY_TEMPLATE = """
Extract the value for '{placeholder}' from this query: '{user_query}'. Only return the value.
"""

DEFAULT_PLACEHOLDER_QUERY = PromptTemplate(
    template=DEFAULT_PLACEHOLDER_QUERY_TEMPLATE,
)
