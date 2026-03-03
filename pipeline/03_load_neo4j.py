"""
Step 3: Load graph data into Neo4j.

Nodes:   Movie, Genre, Tag, User
Edges:   HAS_GENRE, HAS_TAG, RATED

Expects:
  data/processed/movies.csv
  data/processed/tags.csv
  data/processed/ratings_sample.csv

Run: python pipeline/03_load_neo4j.py
"""

import os
import pandas as pd
from neo4j import GraphDatabase
from more_itertools import chunked
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

URI  = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = "neo4j"
PASS = os.getenv("NEO4J_PASS", "moviepass")

driver = GraphDatabase.driver(URI, auth=(USER, PASS))
print(f"Connected to Neo4j: {URI}")


def run_batches(session, query, records, batch_size=500, desc=""):
    batches = list(chunked(records, batch_size))
    for batch in tqdm(batches, desc=desc or query[:40], unit="batch"):
        session.run(query, batch=list(batch))


with driver.session() as s:
    # ── Constraints ───────────────────────────────────────────────────────
    print("Creating constraints...")
    for q in [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Movie) REQUIRE m.movie_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Tag)   REQUIRE t.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User)  REQUIRE u.user_id IS UNIQUE",
    ]:
        s.run(q)
    print("Constraints ready.")

    # ── Movies ────────────────────────────────────────────────────────────
    print("\nLoading Movie nodes...")
    movies = pd.read_csv("data/processed/movies.csv")
    movie_records = [
        {
            "movie_id":    int(r.movie_id),
            "title":       str(r.title),
            "year":        int(r.year) if pd.notna(r.year) else None,
            "bayesian_avg": float(r.bayesian_avg) if pd.notna(r.bayesian_avg) else 0.0,
            "poster_path": str(r.poster_path) if pd.notna(r.poster_path) else "",
        }
        for r in movies.itertuples()
    ]
    run_batches(s, """
        UNWIND $batch AS row
        MERGE (m:Movie {movie_id: row.movie_id})
        SET m.title        = row.title,
            m.year         = row.year,
            m.bayesian_avg = row.bayesian_avg,
            m.poster_path  = row.poster_path
    """, movie_records, desc="Movies")
    print(f"  {len(movie_records):,} Movie nodes loaded.")

    # ── Genres ────────────────────────────────────────────────────────────
    print("\nLoading Genre nodes + HAS_GENRE edges...")
    genre_edges = []
    for r in movies.itertuples():
        raw = str(r.genres) if pd.notna(r.genres) else ""
        for g in raw.split("|"):
            g = g.strip()
            if g and g.lower() != "(no genres listed)":
                genre_edges.append({"movie_id": int(r.movie_id), "genre": g})

    run_batches(s, """
        UNWIND $batch AS row
        MERGE (g:Genre {name: row.genre})
        WITH g, row
        MATCH (m:Movie {movie_id: row.movie_id})
        MERGE (m)-[:HAS_GENRE]->(g)
    """, genre_edges, desc="Genres")
    print(f"  {len(genre_edges):,} genre edges loaded.")

    # ── Tags ──────────────────────────────────────────────────────────────
    print("\nLoading Tag nodes + HAS_TAG edges...")
    tags = pd.read_csv("data/processed/tags.csv")
    valid_ids = set(movies["movie_id"])
    tags = tags[tags["movie_id"].isin(valid_ids)]

    tag_edges = [
        {"movie_id": int(r.movie_id), "tag": str(r.tag), "count": int(r.count)}
        for r in tags.itertuples()
    ]
    run_batches(s, """
        UNWIND $batch AS row
        MERGE (t:Tag {name: row.tag})
        WITH t, row
        MATCH (m:Movie {movie_id: row.movie_id})
        MERGE (m)-[:HAS_TAG {count: row.count}]->(t)
    """, tag_edges, desc="Tags")
    print(f"  {len(tag_edges):,} tag edges loaded.")

    # ── Ratings ───────────────────────────────────────────────────────────
    print("\nLoading User nodes + RATED edges (this is the big one — grab a coffee)...")
    ratings = pd.read_csv("data/processed/ratings_sample.csv")
    ratings = ratings[ratings["movie_id"].isin(valid_ids)]

    rating_records = [
        {"user_id": int(r.user_id), "movie_id": int(r.movie_id), "rating": float(r.rating)}
        for r in ratings.itertuples()
    ]
    run_batches(s, """
        UNWIND $batch AS row
        MERGE (u:User {user_id: row.user_id})
        WITH u, row
        MATCH (m:Movie {movie_id: row.movie_id})
        MERGE (u)-[:RATED {rating: row.rating}]->(m)
    """, rating_records, batch_size=5000, desc="Ratings")
    print(f"  {len(rating_records):,} RATED edges loaded.")

    # ── Sanity check ──────────────────────────────────────────────────────
    print("\nSanity check:")
    for label, q in [
        ("Movie",  "MATCH (m:Movie) RETURN count(m) AS n"),
        ("Genre",  "MATCH (g:Genre) RETURN count(g) AS n"),
        ("Tag",    "MATCH (t:Tag)   RETURN count(t) AS n"),
        ("User",   "MATCH (u:User)  RETURN count(u) AS n"),
        ("RATED",  "MATCH ()-[r:RATED]->() RETURN count(r) AS n"),
    ]:
        n = s.run(q).single()["n"]
        print(f"  {label}: {n:,}")

driver.close()
print("\nNeo4j load complete.")
