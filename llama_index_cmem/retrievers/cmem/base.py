"""llama_index retrievers cmem base"""

import json

from llama_index.core.schema import NodeWithScore, TextNode


def auto_convert_results(
    results: dict, metadata: dict | None = None, score: float = 1.0
) -> list[NodeWithScore]:
    """Convert a results dictionary to a list with a single NodeWithScore object"""
    if metadata is None:
        metadata = {}
    node = TextNode(text=json.dumps(results), metadata=metadata)
    return [NodeWithScore(node=node, score=score)]
