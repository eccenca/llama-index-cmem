"""Example tests.

Remove this and other example files after bootstrapping your project.
"""

from llama_index_cmem.utils.cmem_query import extract_sparql, format_sparql_list

PREDICTION_STR = """
To list all services from the given RDF ontology and limit the results to 20 items,
you can use the following SPARQL query:

```sparql
PREFIX pv: <http://ld.company.org/prod-vocab/>

SELECT ?service ?name
WHERE {
  ?service a pv:Service .
  ?service pv:name ?name .
}
LIMIT 20
```

### Explanation:
- **PREFIX**: This line defines the namespace for the vocabulary used in the query.
In this case, `pv` is defined as `http://ld.company.org/prod-vocab/`.
- **SELECT**: This specifies the variables to return in the results.
Here, `?service` represents the service resource, and `?name` represents the name of the service.
- **WHERE**: This block contains the conditions for the query:
  - `?service a pv:Service .` filters for resources that are of type `Service`.
  - `?service pv:name ?name .` retrieves the name of each service.
- **LIMIT 20**: This limits the number of results returned to 20 items.

You can run this query against the graph located at
`http://ld.company.org/prod-inst/` to get the desired results.
"""

SPARQL_STR = """
PREFIX pv: <http://ld.company.org/prod-vocab/>

SELECT ?service ?name
WHERE {
  ?service a pv:Service .
  ?service pv:name ?name .
}
LIMIT 20
"""

SPARQL_LIST_STR = """
PREFIX pv: <http://ld.company.org/prod-vocab/>

SELECT ?service ?name
WHERE {
  ?service a pv:Service .
  ?service pv:name ?name .
}
LIMIT 20

-----

PREFIX pv: <http://ld.company.org/prod-vocab/>

SELECT ?service ?name
WHERE {
  ?service a pv:Service .
  ?service pv:name ?name .
}
LIMIT 20
"""


def test_extract_sparql() -> None:
    """Test extract sparql"""
    assert extract_sparql(PREDICTION_STR) == SPARQL_STR

def test_format_sparql_list() -> None:
    """Test format sparql list"""
    assert format_sparql_list([SPARQL_STR, SPARQL_STR]) == SPARQL_LIST_STR
