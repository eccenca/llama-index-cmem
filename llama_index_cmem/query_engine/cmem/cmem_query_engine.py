from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.query_engine.custom import STR_OR_RESPONSE_TYPE
from llama_index.core.response_synthesizers import BaseSynthesizer

from llama_index_cmem.retrievers.cmem import CMEMRetriever


class CMEMQueryEngine(CustomQueryEngine):
    """CMEM Query Engine"""

    retriever: CMEMRetriever
    response_synthesizer: BaseSynthesizer

    def custom_query(self, query_str: str) -> STR_OR_RESPONSE_TYPE:
        nodes = self.retriever.retrieve(query_str)
        response_obj = self.response_synthesizer.synthesize(query_str, nodes)
        return response_obj
