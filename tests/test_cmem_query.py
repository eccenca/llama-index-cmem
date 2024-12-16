"""Example tests.

Remove this and other example files after bootstrapping your project.
"""

from llama_index_cmem.utils.cmem_query import extract_sparql, format_sparql_list

PREDICTION_STR = """
Some text before.

```sparql
THIS IS SPARQL
```

And probably more text after.
"""

SPARQL_STR = """THIS IS SPARQL"""

SPARQL_LIST_STR = """THIS IS SPARQL

-----

THIS IS SPARQL"""


def test_extract_sparql() -> None:
    """Test extract sparql"""
    assert extract_sparql(PREDICTION_STR) == SPARQL_STR


def test_format_sparql_list() -> None:
    """Test format sparql list"""
    assert format_sparql_list([SPARQL_STR, SPARQL_STR]) == SPARQL_LIST_STR
