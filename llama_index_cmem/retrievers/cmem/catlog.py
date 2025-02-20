"""Catalog retriever."""

import logging
import re
from typing import Any, cast

from cmem.cmempy.queries import DEFAULT_NS, QUERY_STRING, QueryCatalog, SparqlQuery
from llama_index.core import PromptTemplate, QueryBundle, Settings
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms import LLM
from llama_index.core.prompts import PromptType
from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel, Field

from llama_index_cmem.retrievers.cmem.base import auto_convert_results

DEFAULT_CATALOG_AUTO_SELECT_TEMPLATE = """
You are an expert for SPARQL queries.

Given is a catalog of predefined SPARQL queries.
This query catalog is a list of query objects in JSON format.

Here is an example of a query object:
{
    'identifier': ':unique id',
    'label': 'a label naming the query',
    'description': 'an optional and more detailed description',
    'placeholder': 'an optional list of placeholders',
    'sparql': 'the actual SPARQL query'
}

Look at each query from the catalog and select the best matching query to answer
the user question.

In case the selected query contains a placeholder list, it is necessary
to add meaningful values for each placeholder key.
Look at the user question for useful values or guess a value for each placeholder key.

In case there is no matching query available in the catalog,
return identifier = n/a, no placeholder and found = false.

User question: {query_str}
Query catalog: {catalog_str}
Response:
"""

DEFAULT_CATALOG_AUTO_SELECT_PROMPT = PromptTemplate(
    DEFAULT_CATALOG_AUTO_SELECT_TEMPLATE, prompt_type=PromptType.CUSTOM
)


def get_queries_as_json() -> list[dict[str, Any]]:
    """Get queries as JSON."""
    results = SparqlQuery(QUERY_STRING, query_type="SELECT").get_json_results()
    queries = []
    for result in results["results"]["bindings"]:
        query = {
            "identifier": result.get("query").get("value").replace(DEFAULT_NS, ":"),
            "label": result.get("label").get("value"),
            "description": result.get("description", {}).get("value"),
            "sparql": result.get("text").get("value"),
            "placeholder": re.findall(r"{{([a-zA-Z0-9_-]+)}}", result.get("text").get("value")),
        }
        queries.append(query)
    return queries


def is_valid_query_identifier(identifier: str) -> bool:
    """Check if a query identifier is valid."""
    return identifier.startswith(":")


class Placeholder(BaseModel):
    """Placeholder model."""

    key: str = Field(description="The placeholder key")
    value: str = Field(description="The placeholder value")


class AutoSelect(BaseModel):
    """Auto select model."""

    identifier: str = Field(description="The query identifier", default="")
    placeholder: list[Placeholder] = Field(
        description="A list of placeholders as key-value pairs", default=[]
    )
    found: bool = Field(description="Whether a query was found or not", default=False)


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

    def _auto_select_query(
        self, query_str: str, prompt: PromptTemplate = DEFAULT_CATALOG_AUTO_SELECT_PROMPT
    ) -> AutoSelect:
        queries = get_queries_as_json()
        prediction = self.llm.structured_predict(
            output_cls=AutoSelect, prompt=prompt, query_str=query_str, catalog_str=str(queries)
        )
        return cast(AutoSelect, prediction)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        catalog = QueryCatalog()
        _identifier = self.identifier
        _placeholder = self.placeholder
        results: Any = []
        if _identifier is not None:
            query = catalog.get_query(_identifier)
            metadata = {"cmem": {"identifier": _identifier, "placeholder": _placeholder}}
            if query is None:
                logging.warning(
                    f"CMEM query catalog does not contain a query with identifier "
                    f"'{_identifier}'"
                )
                return auto_convert_results(results, metadata=metadata)
        else:
            query_str = query_bundle.query_str
            auto_select: AutoSelect = self._auto_select_query(query_str)
            _identifier = auto_select.identifier
            _placeholder = {item.key: item.value for item in auto_select.placeholder}
            metadata = {
                "cmem": {
                    "auto_selected_identifier": _identifier,
                    "auto_selected_placeholder": _placeholder,
                }
            }
            if not auto_select.found:
                return auto_convert_results(results, metadata=metadata)
            query = catalog.get_query(_identifier)
        results = query.get_json_results(placeholder=_placeholder)
        return auto_convert_results(results, metadata=metadata)
