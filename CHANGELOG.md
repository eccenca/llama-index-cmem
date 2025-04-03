# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/) and this project adheres to [Semantic Versioning](https://semver.org/)

## [0.6.1] 2025-04-03

### Fixed

- Fix outdated jupyter notebook descriptions
- Update jinja2 dependency to pass safety check

## [0.6.0] 2025-04-01

### Added

- CMEMQueryCatalogRetriever to retrieve nodes from predefined queries
- SPARQLReader to load documents (for ingestion) from SPARQL endpoint
- SPARQLRetriever to retrieve nodes from a SPARQL endpoint
- Some default prompts to work with SPARQL and CMEM query catalog.

### Removed

- solo query builder

### Fixed

- rename CMEMRetriever to NLSPARQLRetriever

## [0.5.0] 2025-01-22

### Added

- example notebooks for solo query builder, query engine and chat engine
- retriever using query builder and graph store to retrieve cmem data as nodes
- query builder with a query object holding predictions and sparql queries
- initial version
