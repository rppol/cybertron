from pydantic import BaseModel, ConfigDict


class Score(BaseModel):
    label: str
    score: float


class ClassifyOut(BaseModel):
    results: list[list[Score]]


class EmbedOut(BaseModel):
    embeddings: list[list[float]]


class Info(BaseModel):
    model_config = ConfigDict(extra="allow")

    precision: str
    inference_precision: str
    hint: str
    streams: int
    threads: int
    optimal_requests: int
    inputs: list[str]
    slots: int
    buckets: list[int]
    tasks: list[str]


class HealthOut(BaseModel):
    status: str
    info: Info


class DeepOut(BaseModel):
    status: str
    seconds: float
