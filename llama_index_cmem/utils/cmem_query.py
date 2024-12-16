"""CMEM query"""

import re

SPARQL_SEPARATOR = "\n-----\n"


def extract_sparql(sparql: str) -> str:
    """Extract SPARQL query with regex."""
    match = re.search(r"(?<=```sparql\n)([\s\S]*?)(?=\n```)", sparql)
    if match:
        return match.group(1)
    return "No SPARQL query found"


def format_sparql_list(sparql_list: list[str]) -> str:
    """Format a list of sparql queries as string"""
    return SPARQL_SEPARATOR.join(sparql_list)


class CMEMQuery:
    """LLM query object"""

    def __init__(self, question: str) -> None:
        self.question: str = question
        self.prediction: list[str] = []
        self.sparql: list[str] = []

    def add(self, prediction: str) -> None:
        """Add prediction"""
        self.prediction.append(prediction)
        extract = extract_sparql(prediction)
        self.sparql.append(extract)

    def get_prediction_list(self) -> list[str]:
        """Get prediction list"""
        return self.prediction

    def get_sparql_list(self) -> list[str]:
        """Get sparql list"""
        return self.sparql

    def get_last_prediction(self) -> str:
        """Get last prediction"""
        return self.prediction[len(self.prediction) - 1]

    def get_last_sparql(self) -> str:
        """Get last sparql"""
        return self.sparql[len(self.sparql) - 1]
