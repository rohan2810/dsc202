## Branch: `caleb` — Change Log

This README documents how the `caleb` branch differs from the state of the starter repo at commit `9251174` (`remove .env from tracking`), which is what I had when I cloned the project.

---

### Summary of changes

- **Step 1 pipeline: resilient Remsky download**
  - In `pipeline/01_download_and_clean.py`, wrapped the HuggingFace `load_dataset` call for the Remsky movie metadata in a retry loop with exponential backoff.
  - On network or connection errors (`requests.ConnectionError` / `OSError`), the script now:
    - Logs a clear message with the current attempt number.
    - Waits (10s → 20s → 40s) between up to 3 attempts.
    - Raises a descriptive `RuntimeError` if all retries fail, instead of failing silently or with a low-level traceback.
  - This makes Step 1 more robust on flaky Wi‑Fi or when HuggingFace is temporarily unavailable, while preserving all original outputs and file formats.

- **Step 1 pipeline: streaming availability (Kaggle)**
  - Step 1 now includes **section 4** that reads `data/raw/MoviesOnStreamingPlatforms.csv` (Kaggle: Netflix, Hulu, Prime Video, Disney+), joins to the canonical `movies` DataFrame on normalized title and year, and writes `data/processed/streaming.csv` (columns: `movie_id`, `provider`).
  - MovieLens titles include a trailing ` (YYYY)`; the join strips that before normalizing so they match Kaggle’s separate Title and Year (e.g. “The Irishman (2019)” matches “The Irishman”, 2019).
  - The merge allows many-to-many on `(title_norm, year)` so remakes or duplicate titles do not cause failures. If the raw CSV is missing, the step skips writing `streaming.csv` and continues.

- **Step 2 pipeline: movie_streaming table**
  - `pipeline/02_load_postgres.py` now creates and loads a **`movie_streaming`** table (`movie_id`, `provider`) from `data/processed/streaming.csv` when that file exists. This does not affect Neo4j or Qdrant; only Postgres is extended.

- **App: streaming in the UI**
  - **Detail sidebar:** When you open “Details / Recommend” for a movie, the sidebar shows **“Where to watch”** with links to each provider’s search (Netflix, Hulu, Prime Video, Disney+) when data exists in `movie_streaming`.
  - **Search and recommendation cards:** The grid of movie cards (search results and “Find Similar” recommendations) is enriched with a batch query to `movie_streaming`. When a movie has streaming data, the card shows a “Watch: Netflix, Prime Video”–style line.
  - Helper `get_streaming_by_movie_ids()` batch-fetches providers from Postgres; the app skips streaming if the `movie_streaming` table is absent.

- **App: Qdrant client compatibility**
  - The app uses a shared helper `_qdrant_vector_search(collection, query_vector, limit)` that calls `qdrant.search(collection_name=..., query_vector=..., limit=...)` when available, otherwise `qdrant.query_points(...).points`, so search and “Find Similar” work across older and newer `qdrant-client` versions.

- **App: Streamlit compatibility**
  - Removed `use_container_width=True` from the sidebar `st.image()` call so the app runs on Streamlit versions that do not support that argument.

- **Step 4 pipeline: Qdrant client compatibility**
  - In `pipeline/04_embed_and_load_qdrant.py`, modified the test search section at the end of the script:
    - The code now prefers `client.search(collection_name=..., query_vector=..., limit=...)` when that API is available.
    - Falls back to `client.query_points(..., query=..., limit=...)` for older client versions.
    - Normalizes the returned results into a `points` list so the subsequent printing logic works across versions.
  - A small debug log block may write to `.cursor/debug-*.log` for local debugging; it does not affect pipeline outputs.

- **App utility: poster URL helper**
  - `poster_url` uses `Optional[str]` and a short docstring; runtime behavior is unchanged.

- **PostgreSQL schema**
  - `docs/postgres-schema.dbml` describes `movies` and `movie_tags`. In code, Step 2 also creates `movie_streaming(movie_id, provider)` with an index on `movie_id`.

- **Python version pin for local tooling**
  - `.python-version` with `3.10` for pyenv etc.; no change to Docker or pipeline behavior.

---

### Behavior notes vs starter repo

- **Pipeline Step 1:** Adds streaming CSV processing (section 4) and more robust Remsky download. Outputs now include `data/processed/streaming.csv` when the Kaggle CSV is present in `data/raw`.
- **Pipeline Step 2:** Adds creation and load of `movie_streaming` from `streaming.csv`; other tables and behavior unchanged.
- **Pipeline Steps 3–4:** Unchanged except Step 4’s test search supports multiple qdrant-client APIs.
- **App:** Search and recommendations work with current qdrant-client and Streamlit; streaming is shown in the sidebar and on cards when `movie_streaming` is populated.
- **Docker compose:** Unchanged from the starter repo at `9251174`.

