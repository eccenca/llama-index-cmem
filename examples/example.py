"""Example module

Remove this and other example files after bootstrapping your project.
"""
from llama_index.llms.openai import OpenAI

from llama_index_cmem.utils.cmem_query_builder import LLMQueryBuilder

from os import environ
# setup the environment for the connection to Corporate Memory
environ["CMEM_BASE_URI"] = ""
environ["OAUTH_GRANT_TYPE"] = "password"
environ["OAUTH_USER"] = "admin"
environ["OAUTH_PASSWORD"] = ""
environ["OAUTH_CLIENT_ID"] = "cmemc"

llm = OpenAI(model="gpt-4o-mini", api_key="")
ontology_graph = "http://ld.company.org/prod-vocab/"
llm_query_builder = LLMQueryBuilder(ontology_graph, llm)
#query = "List all services. Limit the results to 20 items."
query = "List all services with price. Limit the results to 20 items."
context_graph = "<http://ld.company.org/prod-inst/>"
sparql = llm_query_builder.generate_sparql(query, context_graph)
print("Original query:\n" + sparql.original_question)
print("Generated query:\n" + sparql.generated_sparql)
print("SPARQL query:\n" + sparql.get_sparql())