from llama_index.core import Settings
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI

from llama_index_cmem.graph_stores.cmem.cmem_graph_store import CMEMGraphStore
from llama_index_cmem.retrievers.cmem.cmem_retriever import CMEMRetriever

from os import environ
# setup the environment for the connection to Corporate Memory
environ["CMEM_BASE_URI"] = ""
environ["OAUTH_GRANT_TYPE"] = "password"
environ["OAUTH_USER"] = "admin"
environ["OAUTH_PASSWORD"] = ""
environ["OAUTH_CLIENT_ID"] = "cmemc"

llm = OpenAI(model="gpt-4o-mini", api_key="")
Settings.llm = llm

ontology_graph = "http://ld.company.org/prod-vocab/"
context_graph = "<http://ld.company.org/prod-inst/>"

graph_store = CMEMGraphStore()
retriever = CMEMRetriever(graph_store=graph_store,ontology_graph=ontology_graph, context_graph=context_graph, llm=llm)

memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
chat_engine = CondensePlusContextChatEngine(
    retriever=retriever,
    llm=llm,
    memory=memory
)

query = "List all hardware with price. Use uris and labels in the result. Limit the results to 20 items."

print("Query: " + query)
response = chat_engine.stream_chat(query)

#print("Final response: " + str(response))
for token in response.response_gen:
    print(token, end="")


response2 = chat_engine.stream_chat("What else can u tell me about the hardware mentioned?")

#print("Final response: " + str(response))
for token in response2.response_gen:
    print(token, end="")
