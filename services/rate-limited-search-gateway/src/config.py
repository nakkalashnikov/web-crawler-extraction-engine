from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    tavily_api_key: str = "tvly-your_api_key_here"
    tavily_base_url: str = "https://api.tavily.com"

    http_timeout_seconds: float = 30.0

    # Maximum concurrent in-flight requests allowed to upstream
    max_concurrent: int = 10

    # Retry settings on 429 / transport errors
    max_retries: int = 3
    backoff_base_s: float = 1.0

    # Global sliding-window request rate limiter (limiter.py)
    # Caps requests per minute regardless of concurrency
    max_requests_per_minute: int = 60
    rate_limit_pause_seconds: float = 30.0

    retry_wall_clock_cap_seconds: float = 300.0
    log_level: str = "INFO"


settings = Settings()
