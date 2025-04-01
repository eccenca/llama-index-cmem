"""CMEM Query Catalog retriever"""

from cmem.cmempy.queries import QUERY_STRING, SparqlQuery
from llama_index.core import Document, QueryBundle, Settings, VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index_cmem.default_prompts import (
    DEFAULT_PLACEHOLDER_QUERY,
    DEFAULT_SPARQL_QUERY_SELECTION_PROMPT,
)
from llama_index_cmem.executor.cmem_sparql_executor import CMEMSPARQLExecutor
from llama_index_cmem.executor.sparql_query_handler import SPARQLQueryHandler
from llama_index_cmem.readers.sparql_reader import SPARQLReader


class CMEMQueryCatalogRetriever(BaseRetriever):
    """CMEM Query Catalog Retriever"""

    def __init__(
        self,
        llm: LLM | None = None,
        embedding_model: BaseEmbedding | None = None,
        top_k: int = 3,
        fetch_placeholders: bool = True,
    ):
        super().__init__()
        self.top_k = top_k
        self.embedding_model = embedding_model or OpenAIEmbedding()
        self.llm = llm or Settings.llm
        _cmem_sparql_executor = CMEMSPARQLExecutor()
        self._handler = SPARQLQueryHandler(executor=_cmem_sparql_executor)
        self._sparql_reader = SPARQLReader(executor=_cmem_sparql_executor)
        self._fetch_placeholders = fetch_placeholders
        self.query_catalog = self._fetch_query_catalog_from_cmem()
        self._build_index()

    def _fetch_query_catalog_from_cmem(self) -> list[Document]:
        """Retrieve query catalog from CMEM via API."""
        return self._sparql_reader.load_data(
            query=QUERY_STRING,
            doc_id_binding="query",
            text_binding=["label", "description", "text"],
            metadata_binding=["query", "label", "description", "text"],
        )

    def _build_index(self) -> None:
        """Convert queries into embeddings and store them in a vector index."""
        self.index = VectorStoreIndex.from_documents(
            self.query_catalog, embed_model=self.embedding_model
        )
        vector_store = SimpleVectorStore()
        self.retriever = self.index.as_retriever(
            vector_stor=vector_store, similarity_top_k=self.top_k
        )

    def retrieve_query_from_catalog(self, user_query: str) -> list[NodeWithScore]:
        """Retrieve the most relevant SPARQL query from the catalog."""
        retrieved_nodes = self.retriever.retrieve(user_query)

        if not retrieved_nodes:
            return []  # No results found

        # Format retrieved options for LLM
        query_options = "\n".join(
            [f"{i+1}: {node.node.text}" for i, node in enumerate(retrieved_nodes)]  # type: ignore[attr-defined]
        )

        response = self.llm.predict(
            DEFAULT_SPARQL_QUERY_SELECTION_PROMPT,
            user_query=user_query,
            query_options=query_options,
        )
        selected_index = int(response.strip()) - 1  # Convert LLM output to index
        selected_node = retrieved_nodes[selected_index]
        sparql_query = SparqlQuery(selected_node.metadata["text"])
        if self._fetch_placeholders and sparql_query.get_placeholder_keys():
            sparql_query_text = self._parameterize_query(sparql_query, user_query)
            selected_node.metadata["text"] = sparql_query_text
        return [selected_node]

    def _parameterize_query(self, query: SparqlQuery, user_query: str) -> str:
        """Identify and replace placeholders with extracted values from the user query."""
        extracted_values = {}
        for placeholder in query.get_placeholder_keys():
            value = self.llm.predict(
                DEFAULT_PLACEHOLDER_QUERY, placeholder=placeholder, user_query=user_query
            ).strip()
            extracted_values[placeholder] = value

        return str(query.get_filled_text(extracted_values))

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return self.retrieve_query_from_catalog(query_bundle.query_str)
