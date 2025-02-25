"""Catalog retriever."""

import json
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

DEFAULT_AUTO_SELECT_QUERY_TEMPLATE = """
Given is a query catalog as a list of SPARQL query objects in JSON format.
Each query object has an identifier and some optional attributes.

Here is an example of a query object:
```json
{
    'identifier': ':unique id of a query',
    'label': 'a label naming a query',
    'description': 'an optional and more detailed description',
    'sparql': 'the actual SPARQL query'
}
```

Look at each query object from the catalog and select the best matching query to answer
the user question. Return the identifier of the selected query as JSON object like:
```json
{
  "identifier": "query_id"
}
```

In case there is no matching query available in the catalog,
return `null`.

User question: {query_str}
Query catalog: {catalog_str}
Response:
"""

DEFAULT_AUTO_SELECT_QUERY_PROMPT = PromptTemplate(
    DEFAULT_AUTO_SELECT_QUERY_TEMPLATE, prompt_type=PromptType.CUSTOM
)

DEFAULT_AUTO_FILL_PLACEHOLDER_TEMPLATE = """
Given is a list of placeholder keys in JSON format and a user question.

Generate a meaningful value for each placeholder key in the list.
Return the result as JSON dictionary with key-value pairs like:
```json
{
  "key1": "value1",
  "key2": "value2"
}
```

Look at the user question for useful values or guess a value for each placeholder key.
If there are no useful values available, return `null`.

User question: {query_str}
Placeholder keys: {placeholder_keys_str}
Response:
"""

DEFAULT_AUTO_FILL_PLACEHOLDER_PROMPT = PromptTemplate(
    DEFAULT_AUTO_FILL_PLACEHOLDER_TEMPLATE, prompt_type=PromptType.CUSTOM
)


def get_query_catalog_as_json() -> list[dict[str, Any]]:
    """Get query catalog as JSON."""
    results = SparqlQuery(QUERY_STRING, query_type="SELECT").get_json_results()
    queries = []
    for result in results["results"]["bindings"]:
        query = {
            "identifier": result.get("query").get("value").replace(DEFAULT_NS, ":"),
            "label": result.get("label").get("value"),
            "description": result.get("description", {}).get("value"),
            "sparql": result.get("text").get("value"),
        }
        queries.append(query)
    return queries


def find_placeholder_key_in_query(query: SparqlQuery) -> list[str]:
    """Extract query placeholders from SPARQL query."""
    return re.findall(r"{{([a-zA-Z0-9_-]+)}}", query.text)


def is_valid_query_identifier(identifier: str) -> bool:
    """Check if a query identifier is valid."""
    return identifier.startswith(":")


class AutoSelectQuery(BaseModel):
    """Auto select query model."""

    identifier: str = Field(..., description="Unique identifier of a query.")


class AutoFillPlaceholder(BaseModel):
    """Autofill placeholder."""

    data: dict[str, str] = Field(..., description="Placeholder as key-value pair.")


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
        self, query_str: str, prompt: PromptTemplate = DEFAULT_AUTO_SELECT_QUERY_PROMPT
    ) -> AutoSelectQuery | None:
        queries = get_query_catalog_as_json()
        prediction = self.llm.structured_predict(
            output_cls=AutoSelectQuery, prompt=prompt, query_str=query_str, catalog_str=str(queries)
        )
        if prediction == "null":
            return None
        return cast(AutoSelectQuery, prediction)

    def _auto_fill_placeholder(
        self,
        query_str: str,
        placeholder_keys: list[str],
        prompt: PromptTemplate = DEFAULT_AUTO_FILL_PLACEHOLDER_PROMPT,
    ) -> AutoFillPlaceholder | None:
        if not placeholder_keys:
            return None
        placeholder_keys_str = str(placeholder_keys)
        prediction = self.llm.structured_predict(
            output_cls=AutoFillPlaceholder,
            prompt=prompt,
            query_str=query_str,
            placeholder_keys_str=placeholder_keys_str,
        )
        if prediction == "null":
            return None
        return cast(AutoFillPlaceholder, prediction)

    def _default_retrieve(self, catalog: QueryCatalog) -> tuple[dict, dict]:
        metadata = {"cmem": {"identifier": self.identifier, "placeholder": self.placeholder}}
        query = catalog.get_query(self.identifier, self.placeholder)
        if query is None:
            logging.warning(
                f"CMEM query catalog does not contain a query with identifier "
                f"'{self.identifier}'"
            )
            results = {}
        else:
            results = query.get_json_results(placeholder=self.placeholder)
        return results, metadata

    def _auto_retrieve(self, catalog: QueryCatalog, query_str: str) -> tuple[dict, dict]:
        auto_select_query = self._auto_select_query(query_str=query_str)
        if auto_select_query is None:
            logging.warning("Could not find a matching query in CMEM query catalog.")
            metadata = {"cmem": {"identifier": "None", "placeholder": "None"}}
            results = {}
        else:
            query = catalog.get_query(auto_select_query.identifier)
            placeholder_keys = find_placeholder_key_in_query(query)
            if placeholder_keys is None or not placeholder_keys:
                metadata = {
                    "cmem": {"identifier": auto_select_query.identifier, "placeholder": "None"}
                }
                results = query.get_json_results()
            else:
                auto_fill_placeholder = self._auto_fill_placeholder(
                    query_str=query_str, placeholder_keys=placeholder_keys
                )
                if auto_fill_placeholder is None:
                    logging.warning("Could not auto fill placeholders.")
                    metadata = {
                        "cmem": {"identifier": auto_select_query.identifier, "placeholder": "None"}
                    }
                    results = {}
                else:
                    metadata = {
                        "cmem": {
                            "identifier": auto_select_query.identifier,
                            "placeholder": str(auto_fill_placeholder.data),
                        }
                    }
                    results = query.get_json_results(placeholder=auto_fill_placeholder.data)
        return results, metadata

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        catalog = QueryCatalog()
        if self.identifier is not None:
            results, metadata = self._default_retrieve(catalog)
        else:
            results, metadata = self._auto_retrieve(catalog, query_str=query_bundle.query_str)
        return auto_convert_results(results, metadata=metadata)
