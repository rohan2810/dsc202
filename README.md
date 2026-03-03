# CineMatch — Natural Language Movie Finder + Graph Recommender

A multi-database movie recommendation system built for DSC 202 (Data Management for Data Science).

Search for movies using plain English descriptions, click any result, and get graph-based recommendations powered by 25M ratings, semantic embeddings, and user-applied tags.

---

## Architecture

```
User Query
    │
    ▼
Qdrant (vector search)          ← all-mpnet-base-v2 embeddings
    │
    ▼
Streamlit App (app.py)
    │
    ├── PostgreSQL  — movie metadata, cast, director, overview
    ├── Neo4j       — graph: Movie, Genre, Tag, User, RATED edges
    └── Qdrant      — 768-dim vectors for semantic search
```

### Recommendation Signals (composite score)

| Signal | Source | Weight | Description |
|--------|--------|--------|-------------|
| Collaborative filtering | Neo4j | 25% | Movies loved by fans of the seed movie |
| Shared themes | Neo4j (Tags) | 35% | User-applied tags like "cyberpunk", "time travel" |
| Genre overlap | Neo4j (Genres) | 15% | Quality-gated genre matching (avg ≥ 3.5) |
| Semantic similarity | Qdrant | 25% | Nearest neighbors by plot/style embedding |

---

## Datasets

| Dataset | Source | Size |
|---------|--------|------|
| MovieLens 25M | [grouplens.org](https://grouplens.org/datasets/movielens/25m/) | 25M ratings, 62K movies |
| Remsky Ultimate Movies | [HuggingFace](https://huggingface.co/datasets/Remsky/Embeddings__Ultimate_1Million_Movies_Dataset) | 1M movies with overviews, posters, cast |

The pipeline joins these on TMDB ID, producing ~61K movies with full metadata and ~25M ratings.

> **Note:** Raw and processed data files are not committed to git (`.gitignore`). Step 1 of the pipeline downloads everything automatically.

---

## Prerequisites

- Python 3.10+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Postgres, Neo4j, Qdrant)
- ~20 GB free disk space (data + Docker volumes)

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd project
pip install -r requirements.txt
```

### 2. Start the databases

```bash
docker-compose up -d
```

This starts:
- **PostgreSQL** on port 5432
- **Neo4j** on ports 7474 (browser) and 7687 (bolt)
- **Qdrant** on port 6333

Wait ~15 seconds for all services to be healthy before running the pipeline.

### 3. Configure environment

The `.env` file is pre-configured for the Docker setup:

```
QDRANT_HOST=localhost
QDRANT_PORT=6333
NEO4J_URI=bolt://localhost:7687
NEO4J_PASS=moviepass
POSTGRES_DSN=dbname=moviedb user=movie password=movie host=localhost
```

No changes needed unless your ports conflict with existing local services.

### 4. Run the data pipeline

Run the four pipeline steps in order. **Total time: ~2 hours**, mostly step 3.

```bash
# Step 1 — Download & clean (~5 min, downloads ~2GB)
python pipeline/01_download_and_clean.py

# Step 2 — Load PostgreSQL (~1 min)
python pipeline/02_load_postgres.py

# Step 3 — Load Neo4j (~90 min for 25M ratings)
python pipeline/03_load_neo4j.py

# Step 4 — Embed & load Qdrant (~8 min)
python pipeline/04_embed_and_load_qdrant.py
```


> **Note:** Step 3 is the slow one. It loads ~25M RATED edges into Neo4j in batches of 5,000.

### 5. Stop / restart databases

Data is persisted in named Docker volumes and survives restarts:

```bash
# Stop containers (data is safe)
docker-compose stop

# Restart later
docker-compose start

# ⚠️ This deletes ALL data — only if you want a clean slate
docker-compose down -v
```

### 6. Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Deployment

Share a live demo without any cloud setup using ngrok. The databases stay on your local machine.

```bash
# Install ngrok (one-time)
brew install ngrok/ngrok/ngrok

# Add your auth token (get it from https://dashboard.ngrok.com/get-started/your-authtoken)
ngrok config add-authtoken YOUR_TOKEN

# Terminal 1 — start the app
streamlit run app.py

# Terminal 2 — expose publicly
ngrok http 8501
```

ngrok prints a public URL like `https://abc123.ngrok-free.app` — share that with anyone. First-time visitors will see a one-time ngrok warning page; clicking **Visit Site** bypasses it. Works as long as your laptop is on and running.

---

## Usage

**Search** — Type any natural language description in the search bar:
> *"dark psychological thriller with an unreliable narrator"*
> *"feel-good animated film about family and adventure"*
> *"heist movie with a clever twist ending"*

**Recommendations** — Click **Details / Recommend →** on any movie card, then click **Find Similar** in the sidebar. The app traverses the Neo4j graph across 4 signals and shows each card's "Why this?" explanation.

---

## Project Structure

```
project/
├── app.py                          # Streamlit app (all UI + query logic)
├── docker-compose.yml              # Postgres + Neo4j + Qdrant
├── requirements.txt
├── .env                            # DB connection strings
└── pipeline/
    ├── 01_download_and_clean.py    # Download datasets, join, output CSVs
    ├── 02_load_postgres.py         # Load movies + tags into Postgres
    ├── 03_load_neo4j.py            # Build graph (Movie/Genre/Tag/User/RATED)
    └── 04_embed_and_load_qdrant.py # Embed movies, index in Qdrant
```

