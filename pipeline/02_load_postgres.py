"""
Step 2: Load processed CSVs into PostgreSQL.

Expects:
  data/processed/movies.csv
  data/processed/tags.csv

Run: python pipeline/02_load_postgres.py
"""

import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

DSN = os.getenv("POSTGRES_DSN", "dbname=moviedb user=movie password=movie host=localhost")
conn = psycopg2.connect(DSN)
conn.autocommit = False
cur = conn.cursor()

print(f"Connected to Postgres: {DSN}")


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────
cur.execute("""
    DROP TABLE IF EXISTS movie_tags;
    DROP TABLE IF EXISTS movies;

    CREATE TABLE movies (
        movie_id      INTEGER PRIMARY KEY,
        tmdb_id       INTEGER,
        title         TEXT,
        year          INTEGER,
        genres        TEXT,
        overview      TEXT,
        tagline       TEXT,
        director      TEXT,
        top_cast      TEXT,
        poster_path   TEXT,
        vote_average  FLOAT,
        vote_count    INTEGER,
        rating_mean   FLOAT,
        rating_count  INTEGER,
        bayesian_avg  FLOAT
    );

    CREATE TABLE movie_tags (
        movie_id  INTEGER REFERENCES movies(movie_id),
        tag       TEXT,
        count     INTEGER
    );
    CREATE INDEX idx_movie_tags_movie_id ON movie_tags(movie_id);
""")
conn.commit()
print("Schema created.")


# ─────────────────────────────────────────────────────────────────────────────
# Load movies
# ─────────────────────────────────────────────────────────────────────────────
movies = pd.read_csv("data/processed/movies.csv")

COLS = ["movie_id","tmdb_id","title","year","genres","overview","tagline",
        "director","top_cast","poster_path","vote_average","vote_count",
        "rating_mean","rating_count","bayesian_avg"]

# Ensure all expected columns exist
for c in COLS:
    if c not in movies.columns:
        movies[c] = None

# Replace NaN with None for psycopg2
rows = [
    tuple(None if pd.isna(v) else v for v in row)
    for row in movies[COLS].itertuples(index=False)
]

execute_values(
    cur,
    f"INSERT INTO movies ({','.join(COLS)}) VALUES %s ON CONFLICT DO NOTHING",
    rows,
    page_size=500,
)
conn.commit()
print(f"Loaded {len(rows):,} movies into Postgres.")


# ─────────────────────────────────────────────────────────────────────────────
# Load tags
# ─────────────────────────────────────────────────────────────────────────────
tags = pd.read_csv("data/processed/tags.csv")
valid_ids = set(movies["movie_id"])
tags = tags[tags["movie_id"].isin(valid_ids)]

tag_rows = [
    (int(r.movie_id), str(r.tag), int(r.count))
    for r in tags.itertuples(index=False)
]

execute_values(
    cur,
    "INSERT INTO movie_tags (movie_id, tag, count) VALUES %s",
    tag_rows,
    page_size=1000,
)
conn.commit()
print(f"Loaded {len(tag_rows):,} tags into Postgres.")


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
cur.execute("SELECT COUNT(*) FROM movies")
print(f"Verify → movies table: {cur.fetchone()[0]:,} rows")

cur.execute("SELECT COUNT(*) FROM movie_tags")
print(f"Verify → movie_tags table: {cur.fetchone()[0]:,} rows")

cur.execute("""
    SELECT title, year, genres, bayesian_avg
    FROM movies
    ORDER BY bayesian_avg DESC NULLS LAST
    LIMIT 5
""")
print("\nTop 5 movies by Bayesian rating:")
for row in cur.fetchall():
    print(f"  {row[0]} ({row[1]}) | {row[2]} | {row[3]:.2f}")

cur.close()
conn.close()
print("\nPostgres load complete.")
