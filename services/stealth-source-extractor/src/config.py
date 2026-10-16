from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # This worker holds no Tavily credentials — it always calls
    # tavily-search over HTTP for actual content extraction (the one
    # sanctioned worker-calling-worker exception, per Story 2.1).
    tavily_search_url: str = "http://tavily-search:8000"
    yt_dlp_url:        str = "http://yt-dlp:8000"

    # Read/pool timeout of the call to tavily-search. That service queues requests behind its own concurrency
    # semaphore and global rate limiter, so a burst (hundreds of extract_source at once) legitimately waits for
    # minutes; a short timeout here turned the queue into "upstream_unavailable" and lost the source for good.
    tavily_timeout_seconds:  float = 1800.0
    min_text_chars:          int   = 1500   # gates the Wayback-snapshot content check only
    http_timeout_seconds:    float = 25.0
    wayback_timeout_seconds: float = 15.0
    wayback_max_retries:     int   = 2

    browser_user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    browser_accept: str = "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
    browser_accept_language: str = "uk-UA,uk;q=0.9,pl;q=0.8,en-US;q=0.7,en;q=0.6"

    log_level: str = "INFO"

    @property
    def browser_headers(self) -> dict[str, str]:
        return {
            "User-Agent":       self.browser_user_agent,
            "Accept":           self.browser_accept,
            "Accept-Language":  self.browser_accept_language,
        }


settings = Settings()
