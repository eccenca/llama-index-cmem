"""CMEM query engine"""
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer

from llama_index_cmem.retrievers.cmem import CMEMRetriever


class CMEMQueryEngine(CustomQueryEngine):
    """CMEM Query Engine"""

    retriever: CMEMRetriever
    response_synthesizer: BaseSynthesizer

    def custom_query(self, query_str: str) -> RESPONSE_TYPE:
        """Run a custom query with a given retriever and response synthesizer."""
        nodes = self.retriever.retrieve(query_str)
        return self.response_synthesizer.synthesize(query_str, nodes)
