# CineMatch

A movie recommendation system that actually understands what you mean. Type something like *"slow-burn psychological thriller set in a cold city"* and it finds films that match that vibe — not just movies with those words in the title.

Built for DSC 202 using three databases: PostgreSQL for metadata, Neo4j for the graph (25M ratings), and Qdrant for semantic vector search.

---

## How it works

```
User types a query
        │
        ▼
Qdrant ANN search  (query → 768-d vector → cosine similarity)
        │
        ▼
Movie cards  (click any card for details)
        │
        ▼  "Find Similar"
Neo4j graph traversal  +  Qdrant vector search
        │
        ▼
Ranked recommendations with explanations
```

When you click "Find Similar", the app runs four signals and blends them:

| Signal | Source | Weight |
|--------|--------|--------|
| Collaborative filtering | Neo4j | 25% |
| Shared tags | Neo4j | 35% |
| Genre overlap | Neo4j | 15% |
| Semantic similarity | Qdrant | 25% |

---

## Data

| Dataset | Where |
|---------|-------|
| MovieLens 25M | [grouplens.org](https://grouplens.org/datasets/movielens/25m/) — 25M ratings, 62K movies |
| Remsky Ultimate Movies | [HuggingFace](https://huggingface.co/datasets/Remsky/Embeddings__Ultimate_1Million_Movies_Dataset) — overviews, posters, cast |
| MoviesOnStreamingPlatforms | [Kaggle](https://www.kaggle.com/datasets/ruchi798/movies-on-netflix-prime-video-hulu-and-disney) — Netflix/Hulu/Prime/Disney+ flags (optional) |

The first two are joined on TMDB ID to get ~61K movies with full metadata. The Kaggle one is optional — drop `MoviesOnStreamingPlatforms.csv` in `data/raw/` before running the pipeline and the app will show "where to watch" links.

> Raw and processed data files are not in git. Step 1 downloads everything automatically.

---

## Setup

You need Python 3.10+ and Docker Desktop. Budget about 20 GB of disk space.

```bash
git clone <repo-url>
cd project
pip install -r requirements.txt
```

Start the databases:

```bash
docker-compose up -d
```

This starts Postgres (5432), Neo4j (7474/7687), and Qdrant (6333). Wait ~15 seconds before running the pipeline.

The `.env` file is already configured for the Docker setup — no changes needed unless your ports conflict.

---

## Pipeline

Run these in order. Total time is around 2 hours, mostly step 3.

```bash
python pipeline/01_download_and_clean.py   # ~5 min, downloads ~2GB
python pipeline/02_load_postgres.py        # ~1 min
python pipeline/03_load_neo4j.py           # ~90 min (25M ratings)
python pipeline/04_embed_and_load_qdrant.py  # ~8 min
```

Step 3 is the slow one — it loads 25M RATED edges into Neo4j in batches.

---

## Running the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

To stop and restart the databases later:

```bash
docker-compose stop   # data is saved
docker-compose start  # picks up where you left off
docker-compose down -v  # ⚠️ wipes everything
```

---

## Sharing a demo

ngrok lets you share a live link without deploying anything:

```bash
brew install ngrok/ngrok/ngrok
ngrok config add-authtoken YOUR_TOKEN

# terminal 1
streamlit run app.py

# terminal 2
ngrok http 8501
```

ngrok gives you a public URL to share. Works as long as your laptop is running.

---

## Project structure

```
project/
├── app.py
├── docker-compose.yml
├── requirements.txt
├── .env
└── pipeline/
    ├── 01_download_and_clean.py
    ├── 02_load_postgres.py
    ├── 03_load_neo4j.py
    └── 04_embed_and_load_qdrant.py
```
