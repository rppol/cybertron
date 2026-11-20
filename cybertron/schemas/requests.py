from typing import Literal

from pydantic import BaseModel, Field


class TextsIn(BaseModel):
    texts: list[str]
    meta: list[dict] | None = None


class ClassifyIn(TextsIn):
    top_k: int = Field(1, ge=1)


class EmbedIn(TextsIn):
    pooling: Literal["mean", "cls"] = "mean"
    normalize: bool = True
