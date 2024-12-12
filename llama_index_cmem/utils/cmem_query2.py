"""CMEM query 2"""

import re

SPARQL_SEPARATOR = "\n-----\n"


def as_sparql(sparql: str) -> str:
    """Extract SPARQL query with regex."""
    try:
        match = re.search(r"(?<=```sparql\n)([\s\S]*?)(?=\n```)", sparql).group(1)
    except AttributeError:
        match = "No SPARQL query found"
    return match


def format_sparql_list(sparql_list: list[str]) -> str:
    """Format a list of sparql queries as string"""
    return SPARQL_SEPARATOR.join(sparql_list)


class QueryPair:
    """A query pair"""

    def __init__(self, prediction: str, sparql: str) -> None:
        self.prediction = prediction
        self.sparql = sparql

    def get_prediction(self) -> str:
        """Get prediction"""
        return self.prediction

    def get_sparql(self) -> str:
        """Get sparql"""
        return self.sparql


class CMEMQuery2:
    """LLM query object"""

    def __init__(self, question: str) -> None:
        self.question: str = question
        self.prediction: list[str] = []
        self.sparql: list[str] = []
        self.query_list: list[QueryPair] = []

    def add(self, prediction: str) -> None:
        """Add prediction"""
        self.prediction.append(prediction)
        self.sparql.append(as_sparql(prediction))

    def add2(self, prediction: str) -> None:
        """Add prediction"""
        query_pair = QueryPair(prediction, as_sparql(prediction))
        self.query_list.append(query_pair)

    def get_prediction_list(self) -> list[str]:
        """Get prediction list"""
        return self.prediction

    def get_sparql_list(self) -> list[str]:
        """Get sparql list"""
        return self.sparql

    def get_query_list(self) -> list[QueryPair]:
        """Get a list of query pairs."""
        return self.query_list

    def get_last_prediction(self) -> str:
        """Get last prediction"""
        return self.prediction[len(self.prediction) - 1]

    def get_last_sparql(self) -> str:
        """Get last sparql"""
        return self.sparql[len(self.sparql) - 1]

    def get_last_query(self) -> QueryPair:
        """Get last query as QueryPair"""
        return self.query_list[len(self.query_list) - 1]
