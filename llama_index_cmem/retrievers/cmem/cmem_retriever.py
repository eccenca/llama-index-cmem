"""CMEM retriever 2"""

import logging

from llama_index.core import QueryBundle, Settings
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.graph_stores.types import GraphStore
from llama_index.core.llms import LLM
from llama_index.core.schema import NodeWithScore, TextNode

from llama_index_cmem.graph_stores.cmem.cmem_graph_store import is_empty_result
from llama_index_cmem.utils.cmem_query_builder import CMEMQueryBuilder

logger = logging.getLogger(__name__)
DEFAULT_RETRIES = 5


class CMEMRetriever(BaseRetriever):
    """CMEM Retriever"""

    def __init__(
        self,
        graph_store: GraphStore,
        ontology_graph: str,
        context_graph: str,
        llm: LLM,
    ) -> None:
        super().__init__()
        self.graph_store = graph_store
        self.query_builder = CMEMQueryBuilder(
            ontology_graph=ontology_graph, context_graph=context_graph, llm=llm
        )
        self.context_graph = context_graph
        self.llm = llm or Settings.llm

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        cmem_query = self.query_builder.generate_sparql(question=query_bundle.query_str)
        response = self.graph_store.query(query=cmem_query.get_last_sparql())
        index = 0
        while is_empty_result(response) and index < DEFAULT_RETRIES:
            logger.info("Empty CMEM query. Try another query.")
            cmem_query = self.query_builder.refine_sparql(
                question=query_bundle.query_str, cmem_query=cmem_query
            )
            response = self.graph_store.query(query=cmem_query.get_last_sparql())
            index += 1
        metadata = {"cmem_query": cmem_query, "cmem_response": response}
        node = TextNode(text=str(response), metadata=metadata)
        return [NodeWithScore(node=node, score=1.0)]
