"""CMEM retriever"""

import logging

from llama_index.core import QueryBundle, Settings
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.graph_stores.types import GraphStore
from llama_index.core.llms import LLM
from llama_index.core.schema import NodeWithScore, TextNode

from llama_index_cmem.graph_stores.cmem.cmem_graph_store import is_empty_result
from llama_index_cmem.utils.cmem_query_builder import LLMQueryBuilder
from llama_index_cmem.utils.cmem_query_builder2 import CMEMQueryBuilder2

logger = logging.getLogger(__name__)


class CMEMRetriever(BaseRetriever):
    """CMEM Retriever"""

    def __init__(
        self,
        graph_store: GraphStore,
        ontology_graph: str,
        context_graph: str,
        llm: LLM | None = None,
    ) -> None:
        super().__init__()
        self.graph_store = graph_store
        self.query_builder = LLMQueryBuilder(
            ontology_graph=ontology_graph, context_graph=context_graph, llm=llm
        )
        self.query_builder2 = CMEMQueryBuilder2(
            ontology_graph=ontology_graph, context_graph=context_graph, llm=llm
        )
        self.context_graph = context_graph
        self.llm = llm or Settings.llm

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        cmem_query = self.query_builder.generate_sparql(question=query_bundle.query_str)
        sparql = cmem_query.get_sparql()
        response = self.graph_store.query(query=sparql)
        if is_empty_result(response):
            logger.info("Empty CMEM query. Try another query.")
            refined_sparql = self.query_builder.refine_sparql(
                question=query_bundle.query_str, cmem_query=cmem_query
            ).get_refined_sparql()
            response = self.graph_store.query(query=refined_sparql)

        return [NodeWithScore(node=TextNode(text=str(response)), score=1.0)]
