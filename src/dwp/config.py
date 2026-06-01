from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Mode = Literal["realtime", "async", "batch"]
CompletionWindow = Literal["24h", "1h"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    doubleword_api_key: str = Field(..., description="Single Doubleword key. No upstream provider keys.")
    doubleword_base_url: str = "https://api.doubleword.ai/v1"

    mode: Mode = "realtime"
    model_chat: str = "deepseek-ai/DeepSeek-V4-Pro"
    model_judge: str = "deepseek-ai/DeepSeek-V4-Pro"
    max_concurrency: int = 8

    project_name: str = "doubleword-arize"

    phoenix_collector_endpoint: str = "http://localhost:6006"

    batch_completion_window: CompletionWindow = "24h"


settings = Settings()
