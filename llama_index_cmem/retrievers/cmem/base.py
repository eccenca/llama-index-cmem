"""llama_index retrievers cmem base"""

from llama_index.core.schema import NodeWithScore, TextNode


def auto_convert_results(
    results: dict, metadata: dict | None = None, score: float = 1.0
) -> list[NodeWithScore]:
    """Convert a results dictionary to a list of NodeWithScore objects"""
    if metadata is None:
        metadata = {}
    node = TextNode(text=str(results), metadata=metadata)
    return [NodeWithScore(node=node, score=score)]
