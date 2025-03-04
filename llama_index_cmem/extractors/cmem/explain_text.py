"""Explain text extractor"""

from collections.abc import Sequence

from llama_index.core import PromptTemplate, Settings
from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, run_jobs
from llama_index.core.extractors import BaseExtractor
from llama_index.core.llms import LLM
from llama_index.core.schema import BaseNode, MetadataMode
from pydantic import Field, SerializeAsAny

DEFAULT_EXPLAIN_TEMPLATE = """\
Explain the meaning of the given context.
-----
{context_str}
-----
"""


class ExplainTextExtractor(BaseExtractor):
    """Explain text extractor"""

    llm: SerializeAsAny[LLM] = Field(description="The LLM to use for explanation.")
    prompt_template: str = Field(
        default=DEFAULT_EXPLAIN_TEMPLATE, description="Prompt template to use when explaining text."
    )

    def __init__(
        self,
        llm: LLM | None = None,
        prompt_template: str = DEFAULT_EXPLAIN_TEMPLATE,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ) -> None:
        """Init params."""
        super().__init__(
            num_workers=num_workers,
        )
        self.llm = llm or Settings.llm
        self.prompt_template = prompt_template

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "ExplainTextExtractor"

    async def _aexplain_text_from_node(self, node: BaseNode) -> dict[str, str]:
        """Explain text from a node and return its metadata dict."""
        context_str = node.get_content(metadata_mode=MetadataMode.NONE)
        explain_text = await self.llm.apredict(
            PromptTemplate(template=self.prompt_template), context_str=context_str
        )
        return {"explain_text": explain_text.strip()}

    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict]:
        """Explain text from a list of nodes and return its metadata."""
        explain_text_jobs = [self._aexplain_text_from_node(node) for node in nodes]
        metadata_list: list[dict] = await run_jobs(
            explain_text_jobs, show_progress=self.show_progress, workers=self.num_workers
        )
        return metadata_list
