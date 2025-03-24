"""SPARQL Retriever."""

from llama_index.core import BasePromptTemplate
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms import LLM
from llama_index.core.schema import NodeWithScore, QueryBundle, QueryType, TextNode

from llama_index_cmem.default_prompts import DEFAULT_TEXT_TO_SPARQL_PROMPT
from llama_index_cmem.sparql_executor.sparql_executor import SPARQLExecutor
from llama_index_cmem.sparql_executor.sparql_query_handler import SPARQLQueryHandler


class SPARQLRetriever(BaseRetriever):
    """SQL Retriever.

    Retrieves via raw SPARQL statements.

    Args:
        executor (SPARQLExecutor): SPARQL executor.
        return_raw (bool): Whether to return raw results or format results.
            Defaults to True.

    """

    def __init__(
        self,
        executor: SPARQLExecutor,
        return_raw: bool = True,
    ) -> None:
        """Initialize params."""
        self._executor = executor
        self._handler = SPARQLQueryHandler(executor=executor)
        self._return_raw = return_raw
        super().__init__()

    def _format_node_results(self, result: dict) -> list[NodeWithScore]:
        rows = result.get("results", {}).get("bindings", {})
        nodes = []
        for row in rows:
            formated_row = {key: value["value"] for key, value in row.items()}
            nodes.append(
                NodeWithScore(node=TextNode(text=str(formated_row), metadata={"result": row}))
            )
        return nodes

    def retrieve_with_metadata(
        self, str_or_query_bundle: QueryType
    ) -> tuple[list[NodeWithScore], dict]:
        """Retrieve with metadata."""
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        result = self._handler.run_sparql(query_bundle.query_str)
        metadata = {
            "sparql_query": query_bundle.query_str,
            "result": result,
        }
        if self._return_raw:
            return [NodeWithScore(node=TextNode(text=str(result), metadata=metadata))], metadata
        return self._format_node_results(result), metadata

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve nodes given query."""
        retrieved_nodes, _ = self.retrieve_with_metadata(query_bundle)
        return retrieved_nodes


class NLSPARQLRetriever(BaseRetriever):
    """Text-to-SPARQL Retriever.

    Retrieves via text.
    """

    def __init__(  # noqa: PLR0913
        self,
        executor: SPARQLExecutor,
        llm: LLM,
        ontology_triples: str = "",
        return_raw: bool = True,
        text_to_sparql_prompt: BasePromptTemplate | None = None,
        sparql_only: bool = False,
    ):
        """Initialize params."""
        super().__init__()
        self._llm = llm
        self._ontology_triples = ontology_triples
        self._return_raw = return_raw
        self._text_to_sparql_prompt = text_to_sparql_prompt or DEFAULT_TEXT_TO_SPARQL_PROMPT
        self._sparql_only = sparql_only
        self._sparql_retriever = SPARQLRetriever(executor=executor, return_raw=return_raw)

    def retrieve_with_metadata(
        self, str_or_query_bundle: QueryType
    ) -> tuple[list[NodeWithScore], dict]:
        """Retrieve with metadata."""
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        response_str = self._llm.predict(
            self._text_to_sparql_prompt,
            question=query_bundle.query_str,
            ontology_triples=self._ontology_triples,
        )
        sparql_query_str = self._parse_response_to_sparql(response_str)
        if self._sparql_only:
            sparql_only_node = TextNode(text=f"{sparql_query_str}")
            retrieved_nodes = [NodeWithScore(node=sparql_only_node)]
            metadata = {"result": sparql_query_str}
        else:
            retrieved_nodes, metadata = self._sparql_retriever.retrieve_with_metadata(
                sparql_query_str
            )
        return retrieved_nodes, {"sparql_query": sparql_query_str, **metadata}

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve nodes given query."""
        retrieved_nodes, _ = self.retrieve_with_metadata(query_bundle)
        return retrieved_nodes

    @staticmethod
    def _parse_response_to_sparql(response: str) -> str:
        sql_query_start = response.find("SPARQLQuery:")
        if sql_query_start != -1:
            response = response[sql_query_start:]
            if response.startswith("SPARQLQuery:"):
                response = response[len("SPARQLQuery:") :]
        sql_result_start = response.find("SPARQLResult:")
        if sql_result_start != -1:
            response = response[:sql_result_start]
        return response.replace("```sparql", "").replace("```", "").strip()
