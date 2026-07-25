from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import TypedDict

from graphtool.retrieval import SourceReference


class SearchRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["search"] = "search"
    reason: str = Field(min_length=1, pattern=r"\S")
    search_focus: str = Field(min_length=1, pattern=r"\S")


class ExpandRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["expand"] = "expand"
    reason: str = Field(min_length=1, pattern=r"\S")
    source: str = Field(min_length=1, pattern=r"\S")
    chunk_id: str = Field(min_length=1, pattern=r"\S")


RetrievalRecommendation = SearchRecommendation | ExpandRecommendation


class SufficiencyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["conversation", "sufficient", "insufficient"]
    missing_information: list[str] = Field(default_factory=list)
    recommendation: RetrievalRecommendation | None = None

    @field_validator("missing_information")
    @classmethod
    def normalize_missing_information(cls, values: list[str]) -> list[str]:
        normalized = []
        for value in values:
            item = value.strip()
            if item and item not in normalized:
                normalized.append(item)
        return normalized

    @model_validator(mode="after")
    def validate_recommendation(self) -> "SufficiencyDecision":
        if self.verdict != "insufficient":
            if self.missing_information or self.recommendation is not None:
                raise ValueError(
                    "Only insufficient evidence may include missing information "
                    "or a retrieval recommendation."
                )
            return self
        if not self.missing_information:
            raise ValueError(
                "Insufficient evidence requires missing information."
            )
        if self.recommendation is None:
            raise ValueError(
                "Insufficient evidence requires a retrieval recommendation."
            )
        return self


class FinalAnswerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, pattern=r"\S")
    cited_reference_ids: list[str] = Field(default_factory=list)


class ConversationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=8_000, pattern=r"\S")


class QueryDecomposition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subquestions: list[str] = Field(min_length=1, max_length=5)
    knowledge_scope: str | None = None
    unmatched_scope: str = ""

    @field_validator("subquestions")
    @classmethod
    def normalize_subquestions(cls, values: list[str]) -> list[str]:
        normalized = []
        for value in values:
            question = value.strip()
            if question and question not in normalized:
                normalized.append(question)
        if not normalized:
            raise ValueError("At least one non-empty subquestion is required.")
        return normalized

    @model_validator(mode="after")
    def validate_scope_selection(self) -> "QueryDecomposition":
        if self.knowledge_scope is not None:
            self.knowledge_scope = self.knowledge_scope.strip() or None
        self.unmatched_scope = self.unmatched_scope.strip()
        if self.knowledge_scope is not None and self.unmatched_scope:
            raise ValueError(
                "knowledge_scope and unmatched_scope cannot both be set."
            )
        return self


class SubquestionOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    verdict: Literal["sufficient", "insufficient"]
    missing_information: list[str] = Field(default_factory=list)


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    reference: SourceReference


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    source: str
    chunk_id: str
    context_text: str
    reference_ids: list[str] = Field(default_factory=list)
    subquestion_indexes: list[int] = Field(default_factory=list)


class DocumentEvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    source: str
    title: str
    headings: list[str] = Field(default_factory=list)
    reference_ids: list[str] = Field(default_factory=list)
    subquestion_indexes: list[int] = Field(default_factory=list)


class GraphPathEvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    node_ids: list[str]
    edge_ids: list[str]
    chunk_ids: list[str]
    context_text: str
    reference_ids: list[str] = Field(default_factory=list)
    subquestion_indexes: list[int] = Field(default_factory=list)


class AgentChunkReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    source: str
    index: int
    heading_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None


class AgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    status: Literal["complete", "partial"]
    references: list[SourceReference] = Field(default_factory=list)
    search_count: int


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    conversation_summary: str
    question: str
    knowledge_scope: str | None
    subquestions: list[str]
    subquestion_index: int
    subquestion_outcomes: list[SubquestionOutcome]
    evidence: list[EvidenceRecord]
    document_evidence: list[DocumentEvidenceRecord]
    graph_path_evidence: list[GraphPathEvidenceRecord]
    references: list[EvidenceReference]
    search_count: int
    retrieval_count: int
    retrieval_queries: list[str]
    new_evidence_count: int
    duplicate_evidence_count: int
    consecutive_empty_retrievals: int
    allowed_sources: list[str]
    allowed_chunks: list[AgentChunkReference]
    used_neighborhoods: list[str]
    research_action: Literal["tools", "respond", "answer"] | None
    direct_response: str | None
    evaluation: SufficiencyDecision | None
    response: AgentResponse | None
