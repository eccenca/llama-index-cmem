"""Example module

Remove this and other example files after bootstrapping your project.
"""

from llama_index_cmem.utils.cmem_query_builder import download_ontology



from os import environ
# setup the environment for the connection to Corporate Memory
environ["CMEM_BASE_URI"] = ""
environ["OAUTH_GRANT_TYPE"] = "password"
environ["OAUTH_USER"] = "admin"
environ["OAUTH_PASSWORD"] = ""
environ["OAUTH_CLIENT_ID"] = "cmemc"

ontology_graph = "http://ld.company.org/prod-vocab/"

ontology = download_ontology(ontology_graph)
print(ontology.decode("utf-8"))
