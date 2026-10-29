"""Service-level vision model settings, read from ``MVP_*`` environment variables."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from multimodal_vision_pipeline.schemas.extraction import (
    DEFAULT_VISION_MODEL,
    MAX_OUTPUT_TOKENS,
    Effort,
    ExtractionModelConfig,
    VisionModelId,
)


class VisionSettings(BaseSettings):
    """Defaults applied when an ingestion request does not override the extraction config.

    Not strict: environment values arrive as strings and are parsed into the declared
    types. Unknown model ids are still rejected by the ``VisionModelId`` enum.
    """

    model_config = SettingsConfigDict(env_prefix="MVP_", extra="forbid", frozen=True)

    vision_model: VisionModelId = DEFAULT_VISION_MODEL
    vision_max_tokens: int = Field(default=16_000, ge=1, le=MAX_OUTPUT_TOKENS)
    vision_effort: Effort = Effort.HIGH
    vision_request_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    vision_max_retries: int = Field(default=2, ge=0, le=10)
    anthropic_api_key: SecretStr | None = None

    def extraction_config(self) -> ExtractionModelConfig:
        return ExtractionModelConfig(
            model=self.vision_model,
            max_tokens=self.vision_max_tokens,
            effort=self.vision_effort,
            request_timeout_seconds=self.vision_request_timeout_seconds,
            max_retries=self.vision_max_retries,
        )
