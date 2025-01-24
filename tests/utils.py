"""Testing utilities."""

import os
from collections.abc import Generator
from shutil import rmtree
from tempfile import mkdtemp

import pytest
from _pytest.mark import MarkDecorator

from tests.cmemc_command_utils import run, run_without_assertion

needs_cmem: MarkDecorator = pytest.mark.skipif(
    "CMEM_BASE_URI" not in os.environ, reason="Needs CMEM configuration"
)

needs_openai: MarkDecorator = pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Needs OpenAI API key",
)

graphs = {
    "vocab": {
        "location": "https://download.eccenca.com/testing-assets/products-demo-project/prod-vocab.ttl",
        "iri": "http://ld.company.org/prod-vocab/",
    },
    "dataset": {
        "location": "https://download.eccenca.com/testing-assets/products-demo-project/prod-inst.ttl",
        "iri": "http://ld.company.org/prod-inst/",
    },
}


@pytest.fixture
def graph_setup() -> Generator[None, None, None]:
    """Graph setup fixture"""
    # make backup and delete all graphs
    backup_directory = mkdtemp(prefix="cmemc-graphs-backup")
    for _ in graphs.values():
        run_without_assertion(["graph", "export", "--output-dir", backup_directory, _["iri"]])
        run_without_assertion(["graph", "delete", _["iri"]])
        run(["graph", "import", _["location"], _["iri"]])

    yield None
    # remove test graphs
    for _ in graphs.values():
        run(["graph", "delete", _["iri"]])

    # import backup graphs and compare triple counts
    run(["graph", "import", backup_directory])
    rmtree(backup_directory)
