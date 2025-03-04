"""Category extractor"""

from collections.abc import Sequence

from llama_index.core import PromptTemplate, Settings
from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, run_jobs
from llama_index.core.extractors import BaseExtractor
from llama_index.core.llms import LLM
from llama_index.core.schema import BaseNode
from pydantic import Field, SerializeAsAny

DEFAULT_CATEGORY_TEMPLATE = """\
Extract a category for the given context.
-----
{context_str}
-----
"""


class CategoryExtractor(BaseExtractor):
    """Category extractor"""

    llm: SerializeAsAny[LLM] = Field(description="The LLM to use for extraction a category.")
    prompt_template: str = Field(
        default=DEFAULT_CATEGORY_TEMPLATE,
        description="Prompt template to use when extracting a category.",
    )

    def __init__(
        self,
        llm: LLM | None = None,
        prompt_template: str = DEFAULT_CATEGORY_TEMPLATE,
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
        return "CategoryExtractor"

    async def _aextract_category_from_node(self, node: BaseNode) -> dict[str, str]:
        """Extract a category from a node and return its metadata dict."""
        context_str = node.get_content(metadata_mode=self.metadata_mode)
        category = await self.llm.apredict(
            PromptTemplate(template=self.prompt_template), context_str=context_str
        )
        return {"category": category.strip()}

    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict]:
        """Extract a category from a list of nodes and return its metadata."""
        category_jobs = [self._aextract_category_from_node(node) for node in nodes]
        metadata_list: list[dict] = await run_jobs(
            category_jobs, show_progress=self.show_progress, workers=self.num_workers
        )
        return metadata_list
