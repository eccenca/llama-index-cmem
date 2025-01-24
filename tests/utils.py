"""Testing utilities."""

import os

import pytest
from _pytest.mark import MarkDecorator

needs_cmem: MarkDecorator = pytest.mark.skipif(
    "CMEM_BASE_URI" not in os.environ, reason="Needs CMEM configuration"
)

needs_openai: MarkDecorator = pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="Needs OpenAI API key",
)
