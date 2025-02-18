# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/) and this project adheres to [Semantic Versioning](https://semver.org/)

## [Unreleased]

### Added

- catalog retriever allows retrieve nodes from predefined queries
- add cmem reader with a default query for all labels

### FIXED

- catalog:retriever: unknown query identifier does not raise attribute error but returns empty nodes list
- fix ruff linters in example notebooks

## [0.5.0] 2025-01-22

### Added

- example notebooks for solo query builder, query engine and chat engine
- retriever using query builder and graph store to retrieve cmem data as nodes
- query builder with a query object holding predictions and sparql queries
- initial version
