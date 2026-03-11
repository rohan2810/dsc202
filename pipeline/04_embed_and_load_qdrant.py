"""
Step 4: Embed movies with sentence-transformers and load into Qdrant.

Expects: data/processed/movies.csv

Run: python pipeline/04_embed_and_load_qdrant.py
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from dotenv import load_dotenv

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION  = "movies"
BATCH_SIZE  = 256
MODEL_NAME  = "all-mpnet-base-v2"
VECTOR_DIM  = 768

# ─────────────────────────────────────────────────────────────────────────────
# Load model + client
# ─────────────────────────────────────────────────────────────────────────────
print(f"Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
print(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")

# Recreate collection (idempotent for dev)
client.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
)
print(f"Collection '{COLLECTION}' ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Build embedding documents
# ─────────────────────────────────────────────────────────────────────────────
def make_doc(row) -> str:
    """
    Combine overview, tagline, genres, director, title into a single text
    string for embedding. Overview comes first so the model weights semantic
    content over title keywords (avoids literal title matching like "Slow Burn").
    """
    genres   = str(row.get("genres",   "") or "").replace("|", ", ")
    director = str(row.get("director", "") or "")
    overview = str(row.get("overview", "") or "")[:500]
    tagline  = str(row.get("tagline",  "") or "")
    year     = int(row["year"]) if pd.notna(row.get("year")) else ""

    parts = [
        overview,
        f"Tagline: {tagline}"   if tagline  else "",
        f"Genres: {genres}"     if genres   else "",
        f"Director: {director}" if director else "",
        f"{row['title']} ({year})",
    ]
    return " | ".join(p for p in parts if p)


print("Loading movies...")
movies = pd.read_csv("data/processed/movies.csv")
movies = movies.dropna(subset=["overview"])   # skip movies with no description
print(f"Embedding {len(movies):,} movies...")

docs = [make_doc(r) for r in movies.to_dict("records")]


# ─────────────────────────────────────────────────────────────────────────────
# Embed + upsert in batches
# ─────────────────────────────────────────────────────────────────────────────
total_upserted = 0

for i in tqdm(range(0, len(movies), BATCH_SIZE), desc="Embedding + uploading"):
    batch_df   = movies.iloc[i : i + BATCH_SIZE]
    batch_docs = docs[i : i + BATCH_SIZE]

    vectors = model.encode(
        batch_docs,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    points = []
    for row, vec in zip(batch_df.itertuples(index=False), vectors):
        points.append(PointStruct(
            id=int(row.movie_id),
            vector=vec,
            payload={
                "movie_id":    int(row.movie_id),
                "title":       str(row.title),
                "year":        int(row.year)         if pd.notna(row.year)         else None,
                "genres":      str(row.genres)       if pd.notna(row.genres)       else "",
                "director":    str(row.director)     if pd.notna(row.director)     else "",
                "poster_path": str(row.poster_path)  if pd.notna(row.poster_path)  else "",
                "bayesian_avg": float(row.bayesian_avg) if pd.notna(row.bayesian_avg) else 0.0,
            }
        ))

    client.upsert(collection_name=COLLECTION, points=points)
    total_upserted += len(points)

print(f"\nUpserted {total_upserted:,} vectors into Qdrant collection '{COLLECTION}'.")


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
info = client.get_collection(COLLECTION)
print(f"Collection info: {info.points_count:,} points, vector size={VECTOR_DIM}")

print("\nTest search: 'mind-bending sci-fi time travel'")
test_vec = model.encode("mind-bending sci-fi time travel", normalize_embeddings=True).tolist()
# Use search() (query_vector=) when query_points is missing (e.g. qdrant-client 1.9.x)
if hasattr(client, "search"):
    results = client.search(collection_name=COLLECTION, query_vector=test_vec, limit=5)
    points = results if isinstance(results, list) else getattr(results, "points", results)
else:
    results = client.query_points(COLLECTION, query=test_vec, limit=5)
    points = results.points
for h in points:
    print(f"  [{h.score:.3f}] {h.payload['title']} ({h.payload.get('year','')})")

print("\nQdrant load complete.")
