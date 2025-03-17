"""Catalog retriever."""

import logging
from typing import Any, cast

from cmem.cmempy.queries import DEFAULT_NS, QueryCatalog, SparqlQuery
from llama_index.core import (
    PromptTemplate,
    QueryBundle,
    Settings,
    VectorStoreIndex,
)
from llama_index.core.llms import LLM
from llama_index.core.prompts import PromptType
from llama_index.core.vector_stores import SimpleVectorStore
from pydantic import BaseModel, Field

from llama_index_cmem.readers.cmem import CMEMReader
from llama_index_cmem.retrievers.cmem.base import CMEMBaseRetriever

DEFAULT_AUTO_SELECT_QUERY_TEMPLATE = """
Extract the Query that matches the question {query_str} from the following Query Catalog\n
```json
{queries}
```
"""

DEFAULT_AUTO_SELECT_QUERY_PROMPT = PromptTemplate(
    DEFAULT_AUTO_SELECT_QUERY_TEMPLATE, prompt_type=PromptType.CUSTOM
)

DEFAULT_AUTO_FILL_PLACEHOLDER_TEMPLATE = """
Extract the Placeholders from the {query_str} for keys {keys}"""

DEFAULT_AUTO_FILL_PLACEHOLDER_PROMPT = PromptTemplate(
    DEFAULT_AUTO_FILL_PLACEHOLDER_TEMPLATE, prompt_type=PromptType.CUSTOM
)

DEFAULT_VECTOR_STORE_RETRIEVE_TEMPLATE = """
You are given a vector store containing documents, where each document represents a query
with the following structure:

doc id: A unique identifier for the document.
label: A concise title or name of the query.
description: A detailed explanation of the query's purpose.

Task:
Given a user's question, identify the most relevant query from the vector store
and return its document id. If no query is sufficiently relevant, return `null`.

Guidelines:
Evaluate semantic similarity between the user's question and both the label and
description of each document.
Select the query that best matches the user's intent.
If no query meets a reasonable similarity threshold, return `null`.

Input:
User question: {query_str}
Vector store: [List of documents with "doc id", "label", and "description"]

Output:
Return the doc id of the best-matching query.
If no suitable match exists, return `null`.
"""

DEFAULT_VECTOR_STORE_RETRIEVE_PROMPT = PromptTemplate(
    DEFAULT_VECTOR_STORE_RETRIEVE_TEMPLATE, prompt_type=PromptType.CUSTOM
)


def get_query_catalog_as_json() -> list[dict[str, Any]]:
    """Get query catalog as JSON."""
    queries: dict[str, SparqlQuery] = QueryCatalog().get_queries()
    return [
        {
            "identifier": query.url,
            "label": query.label,
            "description": query.description,
        }
        for query in queries.values()
    ]


class SelectQuery(BaseModel):
    """Represents a selectable query with an identifier, label, and description."""

    identifier: str = Field(description="Unique identifier of the query.")
    label: str = Field(default="A label naming a query", description="Short label for the query.")
    description: str = Field(
        default="An optional and more detailed description",
        description="Detailed explanation of the query.",
    )


class Placeholder(BaseModel):
    """Represents a key-value pair for query placeholders."""

    key: str = Field(description="Placeholder key.")
    value: str = Field(description="Placeholder value.")


class Placeholders(BaseModel):
    """Represents a collection of query placeholders."""

    placeholders: list[Placeholder]

    def to_dict(self) -> dict[str, str]:
        """Convert the Placeholders model into a dictionary mapping keys to values."""
        return {placeholder.key: placeholder.value for placeholder in self.placeholders}


class CatalogVectorRetriever(CMEMBaseRetriever):
    """Catalog retriever using vector store."""

    def _retrieve_cmem_results(self, query_bundle: QueryBundle) -> tuple[dict, dict]:
        catalog = QueryCatalog()
        return self._vector_retrieve(catalog, query_bundle.query_str)

    @staticmethod
    def _vector_retrieve(
        catalog: QueryCatalog, query_str: str, index: VectorStoreIndex | None = None
    ) -> tuple[dict, dict]:
        if index is None:
            documents = CMEMReader().load_query_catalog_data()
            vector_store = SimpleVectorStore()
            index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
        query_engine = index.as_query_engine(text_qa_template=DEFAULT_VECTOR_STORE_RETRIEVE_PROMPT)
        response = query_engine.query(str_or_query_bundle=query_str)
        query = response.source_nodes[0].metadata["query"]
        identifier = query.replace(DEFAULT_NS, ":")
        metadata = {"cmem": {"identifier": identifier, "placeholder": "None"}}
        query = catalog.get_query(identifier=identifier)
        results = query.get_json_results()
        return results, metadata


class CatalogAutoSelectRetriever(CMEMBaseRetriever):
    """Catalog Retriever."""

    def __init__(
        self,
        llm: LLM | None = None,
    ) -> None:
        super().__init__()
        self.llm = llm or Settings.llm

    def _auto_select_query(
        self, query_str: str, prompt: PromptTemplate = DEFAULT_AUTO_SELECT_QUERY_PROMPT
    ) -> SelectQuery:
        queries = get_query_catalog_as_json()
        prediction = self.llm.structured_predict(
            output_cls=SelectQuery, prompt=prompt, query_str=query_str, queries=queries
        )
        return cast(SelectQuery, prediction)

    def _auto_fill_placeholder(
        self,
        query_str: str,
        placeholder_keys: list[str],
        prompt: PromptTemplate = DEFAULT_AUTO_FILL_PLACEHOLDER_PROMPT,
    ) -> Placeholders | None:
        if not placeholder_keys:
            return None
        prediction = self.llm.structured_predict(
            output_cls=Placeholders,
            prompt=prompt,
            query_str=query_str,
            keys=",".join(placeholder_keys),
        )
        if prediction == "null":
            return None
        return cast(Placeholders, prediction)

    def _auto_retrieve(self, catalog: QueryCatalog, query_str: str) -> tuple[dict, dict]:
        auto_select_query = self._auto_select_query(query_str=query_str)
        query = catalog.get_query(auto_select_query.identifier)
        if query is None:
            logging.warning("Could not find a matching query in CMEM query catalog.")
            metadata = {"cmem": {"identifier": "None", "placeholder": "None"}}
            results = {}
        else:
            placeholder_keys = query.get_placeholder_keys()
            if not placeholder_keys:
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
                            "placeholder": f"{auto_fill_placeholder.to_dict()}",
                        }
                    }
                    results = query.get_json_results(placeholder=auto_fill_placeholder.to_dict())
        return results, metadata

    def _retrieve_cmem_results(self, query_bundle: QueryBundle) -> tuple[dict, dict]:
        catalog = QueryCatalog()
        return self._auto_retrieve(catalog, query_str=query_bundle.query_str)
