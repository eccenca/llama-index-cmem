"""Pytest configuration."""

import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from attr import dataclass

from tests import FIXTURE_DIR
from tests.cmemc_command_utils import run


@dataclass
class GraphSetup:
    """Graph Setup Dataclass"""

    graphs: dict[str, dict[str, str]]


@pytest.fixture
def graph_setup(tmp_path: Path) -> Generator[GraphSetup, Any, None]:
    """Graph setup fixture"""
    if "CMEM_BASE_URI" not in os.environ:
        pytest.skip("CMEM_BASE_URI not set")
    graph_setup = GraphSetup(
        graphs={
            "vocab": {
                "location": "https://download.eccenca.com/testing-assets/products-demo-project/prod-vocab.ttl",
                "iri": "http://ld.company.org/prod-vocab/",
            },
            "dataset": {
                "location": "https://download.eccenca.com/testing-assets/products-demo-project/prod-inst.ttl",
                "iri": "http://ld.company.org/prod-inst/",
            },
            "combined": {
                "location": "https://download.eccenca.com/testing-assets/products-demo-project/prod-combined.ttl",
                "iri": "http://ld.company.org/prod-combined/",
            },
            "query_catalog": {
                "location": str(FIXTURE_DIR / "query_catalog.ttl"),
                "iri": "https://ns.eccenca.com/data/queries/",
            },
        }
    )
    # make backup and delete all graphs
    store_backup = str(tmp_path / "store.zip")
    run(["admin", "store", "export", store_backup])
    # import our fixture graphs
    for _ in graph_setup.graphs.values():
        run(["graph", "import", "--replace", _["location"], _["iri"]])

    yield graph_setup
    # import backup graphs and compare triple counts
    run(["admin", "store", "import", store_backup])
