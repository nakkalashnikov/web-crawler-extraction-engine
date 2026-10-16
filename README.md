# Distributed High-Throughput Web Crawling & Content Extraction Engine

A production-grade, distributed web crawler and extraction architecture built in Python (FastAPI, `asyncio`, `curl_cffi`, `httpx`).

Engineered specifically to solve large-scale crawling challenges:
- **Rate limiting (429) & quota starvation (432)** across distributed key pools.
- **Bot mitigation bypass** via TLS ClientHello fingerprint impersonation (`curl_cffi`).
- **Multi-stage fallback strategies** (Wayback Machine CDX API, mirror resolvers).
- **Deep unstructured extraction** (HTML normalization, JSON-LD date parsing, captions, and cross-platform video embeds).

---

## 🏗 System Architecture

The architecture consists of two decoupled, high-resilience microservices designed to handle bursts of 100–500+ parallel extraction jobs without dropping requests or triggering ban cascades:

```
                      [ Incoming Crawl / Extract Jobs ]
                                     │
                                     ▼
                ┌─────────────────────────────────────────┐
                │        Stealth Source Extractor         │
                │        (Bypasses TLS Fingerprinting)     │
                └────────────────────┬────────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 │                                       │
                 ▼                                       ▼
    [ Third-Party Target Web ]              [ Rate-Limited Gateway ]
    (curl_cffi Chrome Impersonation)        (Token Bucket / Sliding Window)
    • Direct HTML + JSON-LD                 • Multi-Key Fair Distribution
    • Social Metadata & oEmbed              • Global 429 Penalty Pause
    • Video / Media Embed Signals           • Automatic Fallback / Retries
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     │
                                     ▼
                      [ Wayback CDX Fallback Engine ]
                      (Recovers 403 / Defunct / Banned URLs)
```

---

## 🚀 Key Modules & Deep-Dive

### 1. `Rate-Limited Search & Extract Gateway`
Located in `services/rate-limited-search-gateway/`:
* **Sliding Window Request Limiter (`limiter.py`):**
  * Tracks exact request timestamps over a monotonic clock window using an efficient `collections.deque`.
  * Guarantees hard throughput limits (e.g., max 60 req/min) across arbitrary concurrency levels.
* **429 Global Pause (`penalize()`):**
  * Parses upstream `Retry-After` headers and pauses **all** concurrent callers simultaneously.
  * Solves the **thundering herd / retry storm problem**, preventing IP-level or account-level bans.
* **Weighted Multi-Key Pool with Concurrency Semaphores (`keys.py`, `search.py`):**
  * Individual `asyncio.Semaphore` per API key, held continuously across exponential backoffs.
  * Separate handling for transient throttles (429: backoff with jitter on same key) vs permanent quota exhaustion (432: immediate rotation to next tier).

### 2. `Stealth Source Extractor`
Located in `services/stealth-source-extractor/`:
* **TLS ClientHello Impersonation (`curl_cffi`):**
  * Major CDNs (Cloudflare, Akamai) detect and block standard Python HTTP clients (`requests`, `httpx`) via JA3/JA4 fingerprints alone.
  * Uses `curl_cffi` with `impersonate="chrome"` for raw third-party HTML fetching.
* **Dual-Path Parallel Extraction (`extract.py`):**
  * Runs upstream extraction and local DOM/embed parsing concurrently with `asyncio.gather()`.
  * Extracts JSON-LD schema metadata, figure `<figcaption>` contexts, OG tags, and publication timestamps without heavy headless browsers.
* **Archival Fallback (`_try_wayback`):**
  * Automatically recovers content from the Internet Archive Wayback CDX API with retry ladders against transient empty snapshot states.
* **Media & Embed Signal Detection (`embed_parser.py`, `indirect_video_signals.py`):**
  * Regex and DOM parsing for YouTube, Twitter/X, TikTok, Facebook, Vimeo, and direct video manifests (HLS/DASH).

---

## 🛠 Tech Stack

* **Language:** Python 3.11+
* **Framework:** FastAPI, Pydantic v2
* **Networking & HTTP:** `httpx` (async client), `curl_cffi` (stealth TLS impersonation)
* **Concurrency:** `asyncio` (queues, locks, semaphores, sliding window limiters)
* **Testing:** `pytest`, `pytest-asyncio`, `respx`

---

## 🚦 Getting Started

### 1. Run with Docker Compose
```bash
docker compose up --build
```

### 2. Run Tests
```bash
# Rate-Limited Gateway tests
cd services/rate-limited-search-gateway
pytest

# Stealth Extractor tests
cd ../stealth-source-extractor
pytest
```
