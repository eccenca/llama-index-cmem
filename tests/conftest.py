"""Pytest configuration."""

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from attr import dataclass

from tests.cmemc_command_utils import run, run_without_assertion


@dataclass
class GraphSetup:
    """Graph Setup Dataclass"""

    graphs: dict[str, dict[str, str]]


@pytest.fixture
def graph_setup(tmp_path: Path) -> Generator[GraphSetup, Any, None]:
    """Graph setup fixture"""
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
        }
    )
    # make backup and delete all graphs
    for _ in graph_setup.graphs.values():
        run_without_assertion(["graph", "export", "--output-dir", str(tmp_path), _["iri"]])
        run_without_assertion(["graph", "delete", _["iri"]])
        run(["graph", "import", _["location"], _["iri"]])

    yield graph_setup
    # remove test graphs
    for _ in graph_setup.graphs.values():
        run(["graph", "delete", _["iri"]])

    # import backup graphs and compare triple counts
    run(["graph", "import", str(tmp_path), "vocab"])
