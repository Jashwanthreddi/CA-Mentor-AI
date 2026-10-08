from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question asked by the student.",
        examples=["What is accounting?"],
    )


class ChatSource(BaseModel):
    chunk_id: str
    rank: int
    reranker_score: float
    page_start: int
    page_end: int


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[ChatSource]