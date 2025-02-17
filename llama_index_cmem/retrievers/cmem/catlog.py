"""Catalog retriever."""

from cmem.cmempy.queries import QueryCatalog
from llama_index.core import QueryBundle, Settings
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms import LLM
from llama_index.core.schema import NodeWithScore

from llama_index_cmem.retrievers.cmem.base import auto_convert_results


class CatalogRetriever(BaseRetriever):
    """Catalog Retriever."""

    def __init__(
        self,
        identifier: str | None = None,
        placeholder: dict | None = None,
        llm: LLM | None = None,
    ) -> None:
        super().__init__()
        self.identifier = identifier
        self.placeholder = placeholder
        self.llm = llm or Settings.llm

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        catalog = QueryCatalog()
        if self.identifier is not None:
            query = catalog.get_query(self.identifier, self.placeholder)
        else:
            query_str = query_bundle.query_str
            query = catalog.get_query(query_str)
        results = query.get_json_results(placeholder=self.placeholder)
        return auto_convert_results(results)
