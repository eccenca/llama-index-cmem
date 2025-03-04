"""Key-value details extractor"""

from collections.abc import Sequence

from llama_index.core import PromptTemplate, Settings
from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, run_jobs
from llama_index.core.extractors import BaseExtractor
from llama_index.core.llms import LLM
from llama_index.core.schema import BaseNode
from pydantic import BaseModel, Field, SerializeAsAny

DEFAULT_KEY_VALUE_DETAILS_TEMPLATE = """\
Extract all details as list of key value pairs from the given context.
-----
{context_str}
-----
"""


class Detail(BaseModel):
    """Detail model"""

    key: str = Field(description="Extracted key.")
    value: str = Field(description="Extracted value.")


class Details(BaseModel):
    """Details model"""

    details: list[Detail] = Field(description="A list of all details.")


class KeyValueDetailsExtractor(BaseExtractor):
    """Detail key/value extractor"""

    llm: SerializeAsAny[LLM] = Field(description="The LLM to use for extraction details.")
    prompt_template: str = Field(
        default=DEFAULT_KEY_VALUE_DETAILS_TEMPLATE,
        description="Prompt template to use when extracting details.",
    )

    def __init__(
        self,
        llm: LLM | None = None,
        prompt_template: str = DEFAULT_KEY_VALUE_DETAILS_TEMPLATE,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ) -> None:
        """Init params."""
        super().__init__(num_workers=num_workers)
        self.llm = llm or Settings.llm
        self.prompt_template = prompt_template

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "KeyValueDetailsExtractor"

    async def _aextract_structured_details_from_node(self, node: BaseNode) -> dict[str, BaseModel]:
        """Extract details from a node and return its metadata dict."""
        context_str = node.metadata["explain_text"]
        details = await self.llm.astructured_predict(
            Details, PromptTemplate(template=self.prompt_template), context_str=context_str
        )
        return {"details": details}

    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict]:
        """Extract details from a list of nodes and return its metadata."""
        details_jobs = [self._aextract_structured_details_from_node(node) for node in nodes]
        metadata_list: list[dict] = await run_jobs(
            details_jobs, show_progress=self.show_progress, workers=self.num_workers
        )
        return metadata_list
