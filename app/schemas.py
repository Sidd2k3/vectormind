"""Request bodies. Pydantic validates them, so bad input is rejected with
a clear 422 error before any handler runs."""

from pydantic import BaseModel, Field


class InsertItemRequest(BaseModel):
    text: str = Field(min_length=1, max_length=300)
    category: str = Field(default="custom", min_length=1, max_length=30)
    # Optional: if omitted, the text is encoded with the demo keyword encoder.
    embedding: list[float] | None = None


class DocInsertRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=500_000)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=3, ge=1, le=10)
