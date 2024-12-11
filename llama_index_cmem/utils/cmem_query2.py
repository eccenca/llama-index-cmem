import re
from typing import List

SPARQL_SEPARATOR = "\n-----\n"

def as_sparql(sparql: str) -> str:
    """Extract pure SPARQL query

    Returns:
    The pure SPARQL query without explanation.

    """

    try:
        match = re.search(r"(?<=```sparql\n)([\s\S]*?)(?=\n```)", sparql).group(1)
    except AttributeError:
        match = "No SPARQL query found"
    return match

def format_sparql_list(sparql_list: List[str]) -> str:
    """Format a list of sparql queries as string"""

    return SPARQL_SEPARATOR.join(sparql_list)

class CMEMQuery2:
    """LLM query object.

        This query object holds the original query together with the generated queries of the LLM.

        Args:
            question (str): The original query.
        """

    def __init__(
            self,
            question
    ) -> None:
        self.question: str = question
        self.prediction: List[str] = []
        self.sparql: List[str] = []
        self.query_list: List[QueryPair] = []

    def add(self, prediction: str):
        self.prediction.append(prediction)
        self.sparql.append(as_sparql(prediction))

    def add2(self, prediction: str):
        query_pair = QueryPair(prediction,as_sparql(prediction))
        self.query_list.append(query_pair)

    def get_prediction_list(self):
        return self.prediction

    def get_sparql_list(self):
        return self.sparql

    def get_query_list(self):
        return self.query_list

    def get_last_prediction(self):
        return self.prediction[len(self.prediction) - 1]

    def get_last_sparql(self):
        return self.sparql[len(self.sparql) - 1]

    def get_last_query(self):
        return self.query_list[len(self.query_list) - 1]

class QueryPair:
    """A query pair"""

    def __init__(
            self,
            prediction: str,
            sparql: str
    ) -> None:
        self.prediction = prediction
        self.sparql = sparql

    def get_prediction(self):
        return self.prediction

    def get_sparql(self):
        return self.sparql