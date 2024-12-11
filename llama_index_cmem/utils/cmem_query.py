import re


def as_sparql(sparql: str) -> str:
    """Get pure SPARQL query

    Returns:
    The pure SPARQL query without explanation.

    """
    try:
        match = re.search(r"(?<=```sparql\n)([\s\S]*?)(?=\n```)", sparql).group(1)
    except AttributeError:
        match = "No SPARQL query found"
    return match


class CMEMQuery:
    """LLM query object.

    This query object holds the original query together with the generated queries of the LLM.

    Args:
        original_question (str): The original query.

    """

    def __init__(self, original_question) -> None:
        self.original_question = original_question
        self.sparql_prediction = None
        self.sparql = None
        self.refined_prediction = None
        self.refined_sparql = None

    def set_sparql_prediction(self, sparql_prediction):
        self.sparql_prediction = sparql_prediction

    def get_sparql(self):
        if self.sparql is None:
            self.sparql = as_sparql(self.sparql_prediction)
        return self.sparql

    def set_refined_prediction(self, refined_prediction):
        self.refined_prediction = refined_prediction

    def get_refined_sparql(self):
        if self.refined_sparql is None:
            self.refined_sparql = as_sparql(self.refined_prediction)
        return self.refined_sparql
