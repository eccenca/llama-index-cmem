"""Example module

Remove this and other example files after bootstrapping your project.
"""
from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI

from llama_index_cmem.graph_stores.cmem import CMEMGraphStore
from llama_index_cmem.retrievers.cmem import CMEMRetriever
from llama_index_cmem.retrievers.cmem.cmem_retriever2 import CMEMRetriever2


def double(value: int) -> int:
    """Get a value"""
    return value * 2

from os import environ
# setup the environment for the connection to Corporate Memory
environ["CMEM_BASE_URI"] = ""
environ["OAUTH_GRANT_TYPE"] = "password"
environ["OAUTH_USER"] = "admin"
environ["OAUTH_PASSWORD"] = ""
environ["OAUTH_CLIENT_ID"] = "cmemc"

llm = OpenAI(model="gpt-4o-mini", api_key="")
Settings.llm = llm

response_synthesizer = get_response_synthesizer()
#response_synthesizer = get_response_synthesizer(streaming=True)

ontology_graph = "http://ld.company.org/prod-vocab/"
context_graph = "<http://ld.company.org/prod-inst/>"

graph_store = CMEMGraphStore()
retriever = CMEMRetriever2(graph_store=graph_store,ontology_graph=ontology_graph, context_graph=context_graph, llm=llm)
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

#query = "List all services with price. Limit the results to 20 items."
#query = "List all hardware with price. Use uris and labels in the result. Limit the results to 20 items."
#query = "What is the hardware with the highest price? What does the hardware costs? Return all uris and labels."
#query = "What is the service with the highest price? Return all uris and labels."
#query = "What is the most expensive service and what does it costs? Return all uris and labels."

#query = "What is the label and price amount and currency of the most expensive hardware product?"
#query = "List all hardware and the corresponding product manager. Limit the results to 20 items."
#query = "What is the name, price and currency of the most expensive service product?"
#query = "How many service and hardware products (combined) do we have?"
#query = "Which product manager has the manages the most products?"
query = "What is the product category where the most products manager are experts in?"
#query = "What is the product category where the least products manager are experts in?"

#query = "What is the product category with the most persons as experts? Consider that instances of such experts can exist in agent and all it's subclasses."

#query = "What are the labels and price amount and currency of the most expensive hardware products?"

#query = "Which `pv:Product` is associated with Network through the `pv:hasCategory` property?"

print("Query: " + query)
response = query_engine.query(query)

print("Final response: " + str(response))
#response.print_response_stream()
