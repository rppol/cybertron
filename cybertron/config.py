from pathlib import Path
from typing import Literal

from pydantic import Field, PositiveInt, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

OWNED = {"PERFORMANCE_HINT", "NUM_STREAMS", "INFERENCE_NUM_THREADS", "ENABLE_HYPER_THREADING", "ENABLE_CPU_PINNING", "CACHE_DIR"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CYBERTRON_")

    model: Path
    labels: list[str] | None = None
    hooks: str | None = None

    max_len: int = Field(128, ge=2)
    buckets: list[PositiveInt] = [16, 32, 64, 128]
    batch_size: PositiveInt = 32

    precision: Literal["FP32", "FP16", "INT8"] = "FP32"
    hint: Literal["LATENCY", "THROUGHPUT"] = "THROUGHPUT"
    streams: PositiveInt | None = None
    threads: PositiveInt | None = None
    hyper_threading: bool | None = None
    cpu_pinning: bool | None = None
    ov_config: dict[str, str] = {}
    cache_dir: Path | None = None

    slots: int = Field(0, ge=0)
    max_texts: PositiveInt = 256
    max_chars: PositiveInt = 4096
    api_key: SecretStr | None = None
    rate_limit: int = Field(0, ge=0)
    rate_window: float = Field(60.0, gt=0)

    gc_interval: float = Field(300.0, ge=0)
    gc_thresholds: tuple[int, int, int] = (50_000, 20, 20)

    @field_validator("ov_config")
    @classmethod
    def _free_keys(cls, v):
        if taken := sorted(OWNED & v.keys()):
            raise ValueError(f"set {taken} through their own settings, not ov_config")
        return v

    @model_validator(mode="after")
    def _cap_buckets(self):
        self.buckets = sorted({b for b in self.buckets if b < self.max_len}) + [self.max_len]
        return self
