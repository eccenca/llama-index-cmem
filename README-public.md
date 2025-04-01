# llama-index-cmem

llama-index tools eccenca Corporate Memory Integration

  
[![poetry][poetry-shield]][poetry-link] [![ruff][ruff-shield]][ruff-link] [![mypy][mypy-shield]][mypy-link] [![copier][copier-shield]][copier] 

## Usage

The llama-index-cmem package allows using [eccenca Corporate Memory](https://eccenca.com/products/enterprise-knowledge-graph-platform-corporate-memory) together with [LlamaIndex](https://docs.llamaindex.ai/en/stable/).

### Components
- SPARQLReader: Use to load documents from SPARQL query to use in ingestion pipeline.
- SPARQLRetriever: Execute SPARQL query to retrieve context from SPARQL endpoint.
- NLSPARQLRetriever: Text-to-SPARQL retriever to retrieve context from SPARQL endpoint using natural language.
- CMEMQueryCatalogRetriever: Execute SPARQL query from CMEM query catalog to retrieve context from SPARQL endpoint.

There are several example notebooks available:
- NLSPARQLRetriever and CMEMQueryCatalogRetriever.
- ChatEngine: Chatting with an RDF knowledge graph data.
- QueryEngine: Single Q&A engine with RDF knowledge graph data.
- PGVector: Generate embeddings from RDF knowledge graph and query with natural language (via vector similarity).


[poetry-link]: https://python-poetry.org/
[poetry-shield]: https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json
[ruff-link]: https://docs.astral.sh/ruff/
[ruff-shield]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&label=Code%20Style
[mypy-link]: https://mypy-lang.org/
[mypy-shield]: https://www.mypy-lang.org/static/mypy_badge.svg
[copier]: https://copier.readthedocs.io/
[copier-shield]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json

