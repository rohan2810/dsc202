"""
Step 1: Download data and produce clean CSVs.

Outputs:
  data/processed/movies.csv        - canonical movie records (~25-35k)
  data/processed/tags.csv          - movie_id, tag, count
  data/processed/ratings_sample.csv - movie_id, user_id, rating (all eligible ~25M)

Run: python pipeline/01_download_and_clean.py
"""

import os
import io
import time
import zipfile
import requests
import pandas as pd
from tqdm import tqdm

RAW  = "data/raw"
PROC = "data/processed"
os.makedirs(f"{RAW}/ml-25m", exist_ok=True)
os.makedirs(PROC, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Download MovieLens 25M
# ─────────────────────────────────────────────────────────────────────────────
ML_ZIP = f"{RAW}/ml-25m.zip"
ML_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"

if not os.path.exists(f"{RAW}/ml-25m/movies.csv"):
    print("Downloading MovieLens 25M (~250 MB)...")
    r = requests.get(ML_URL, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(ML_ZIP, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    print("Extracting...")
    with zipfile.ZipFile(ML_ZIP) as z:
        z.extractall(RAW)
    print("MovieLens extracted.")
else:
    print("MovieLens already downloaded, skipping.")

ml_movies  = pd.read_csv(f"{RAW}/ml-25m/movies.csv")
ml_links   = pd.read_csv(f"{RAW}/ml-25m/links.csv")
ml_ratings = pd.read_csv(f"{RAW}/ml-25m/ratings.csv")
ml_tags    = pd.read_csv(f"{RAW}/ml-25m/tags.csv")

print(f"MovieLens: {len(ml_movies):,} movies, {len(ml_ratings):,} ratings, {len(ml_tags):,} tags")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load Remsky metadata from HuggingFace
# ─────────────────────────────────────────────────────────────────────────────
REMSKY_CACHE = f"{RAW}/remsky_movies.parquet"

if not os.path.exists(REMSKY_CACHE):
    print("Loading Remsky movie metadata from HuggingFace (may take a few minutes)...")
    from datasets import load_dataset
    max_retries, backoff = 3, 10
    for attempt in range(max_retries):
        try:
            remsky_ds = load_dataset(
                "Remsky/Embeddings__Ultimate_1Million_Movies_Dataset",
                split="train",
                streaming=False,
            )
            remsky = remsky_ds.to_pandas()
            remsky.to_parquet(REMSKY_CACHE, index=False)
            print(f"Remsky: {len(remsky):,} movies cached to {REMSKY_CACHE}")
            break
        except (requests.exceptions.ConnectionError, OSError) as e:
            if attempt < max_retries - 1:
                print(f"HuggingFace connection failed (attempt {attempt + 1}/{max_retries}), retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
            else:
                raise RuntimeError(
                    "Could not load Remsky dataset from HuggingFace after retries. "
                    "Check your network or try again later."
                ) from e
else:
    print("Loading Remsky from cache...")
    remsky = pd.read_parquet(REMSKY_CACHE)
    print(f"Remsky: {len(remsky):,} movies")

# Keep only the columns we need
REMSKY_COLS = ["id", "imdb_id", "title", "overview", "tagline",
               "genres", "movie_cast", "director", "poster_path",
               "vote_average", "vote_count", "release_date", "runtime"]
remsky = remsky[[c for c in REMSKY_COLS if c in remsky.columns]].copy()
remsky = remsky.rename(columns={"id": "tmdb_id"})
remsky["tmdb_id"] = pd.to_numeric(remsky["tmdb_id"], errors="coerce")
remsky = remsky.dropna(subset=["tmdb_id"])
remsky["tmdb_id"] = remsky["tmdb_id"].astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Join MovieLens ↔ Remsky via tmdbId
# ─────────────────────────────────────────────────────────────────────────────
print("Joining datasets...")

links = ml_links.dropna(subset=["tmdbId"]).copy()
links["tmdbId"] = links["tmdbId"].astype(int)

# Bayesian average rating (more robust than raw mean for ranking)
stats = ml_ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
stats.columns = ["movieId", "rating_mean", "rating_count"]
C = stats["rating_mean"].mean()
m = stats["rating_count"].quantile(0.25)
stats["bayesian_avg"] = (
    (stats["rating_count"] / (stats["rating_count"] + m)) * stats["rating_mean"]
    + (m / (stats["rating_count"] + m)) * C
)

movies = (
    ml_movies                                                           # movieId, title, genres
    .merge(links, on="movieId")                                         # + tmdbId, imdbId
    .merge(remsky, left_on="tmdbId", right_on="tmdb_id",
           how="inner", suffixes=("_ml", "_tmdb"))                      # + overview, poster, ...
    .merge(stats, on="movieId", how="left")                            # + rating stats
)

# Canonical column cleanup
# title_ml comes from MovieLens (cleaner, already in English)
# title_tmdb comes from Remsky; use ml title as primary
movies = movies.rename(columns={
    "movieId":    "movie_id",
    "title_ml":   "title",
    "genres_ml":  "genres",          # pipe-separated: "Action|Drama"
    "tmdbId":     "tmdb_id_link",
})
movies["year"] = pd.to_datetime(movies["release_date"], errors="coerce").dt.year.astype("Int64")

# Trim cast to top 5 names (it may be a comma-separated string)
def trim_cast(val):
    if pd.isna(val) or str(val).strip() == "":
        return ""
    names = [n.strip() for n in str(val).split(",")]
    return ", ".join(names[:5])

movies["top_cast"] = movies["movie_cast"].apply(trim_cast)

final_cols = [
    "movie_id", "tmdb_id", "title", "year", "genres",
    "overview", "tagline", "director", "top_cast",
    "poster_path", "vote_average", "vote_count",
    "rating_mean", "rating_count", "bayesian_avg",
]
movies = movies[[c for c in final_cols if c in movies.columns]].drop_duplicates("movie_id")

movies.to_csv(f"{PROC}/movies.csv", index=False)
print(f"Saved {len(movies):,} movies → {PROC}/movies.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Streaming availability (Kaggle MoviesOnStreamingPlatforms)
# ─────────────────────────────────────────────────────────────────────────────
# Build a clean streaming availability file by joining the Kaggle
# MoviesOnStreamingPlatforms dataset to the canonical movies DataFrame.
streaming_raw_path = f"{RAW}/MoviesOnStreamingPlatforms.csv"
streaming_out_path = f"{PROC}/streaming.csv"

if os.path.exists(streaming_raw_path):
    print("Building streaming availability CSV from MoviesOnStreamingPlatforms.csv...")
    streaming_raw = pd.read_csv(streaming_raw_path)

    # Normalize title and year in the Kaggle dataset for a robust join.
    streaming_raw["title_norm"] = (
        streaming_raw["Title"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    streaming_raw["year"] = pd.to_numeric(
        streaming_raw["Year"], errors="coerce"
    ).astype("Int64")

    # Build a slim join frame from canonical movies with the same normalized key.
    # Strip trailing " (YYYY)" from MovieLens titles so they match Kaggle (Title + Year).
    movies_join = movies[["movie_id", "title", "year"]].copy()
    title_clean = (
        movies_join["title"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
        .str.strip()
    )
    movies_join["title_norm"] = title_clean.str.lower()

    # Provider columns in the streaming dataset that indicate availability as 0/1.
    provider_cols = ["Netflix", "Hulu", "Prime Video", "Disney+"]
    for col in provider_cols:
        if col not in streaming_raw.columns:
            streaming_raw[col] = 0

    # Reshape to long format so each row represents one (movie, provider) combination.
    streaming_long = streaming_raw.melt(
        id_vars=["title_norm", "year"],
        value_vars=provider_cols,
        var_name="provider",
        value_name="available",
    )

    # Keep only rows where the movie is available on that provider.
    streaming_long = streaming_long[streaming_long["available"] == 1].copy()
    if streaming_long.empty:
        print("No streaming availability rows after filtering; skipping streaming.csv.")
    else:
        # Normalize provider names (for example, "Prime Video" → "prime video").
        streaming_long["provider"] = (
            streaming_long["provider"].astype(str).str.strip().str.lower()
        )

        # #region agent log
        _key_counts = movies_join.groupby(["title_norm", "year"]).size()
        _dupes = _key_counts[_key_counts > 1]
        import json
        _log = open("/Users/caleb/Desktop/Winter-26-Classes/202/dsc202/.cursor/debug-8ecafa.log", "a")
        _log.write(json.dumps({"sessionId":"8ecafa","hypothesisId":"H1","location":"01_download_and_clean.py:streaming_merge","message":"movies_join merge key uniqueness","data":{"movies_join_rows":len(movies_join),"unique_title_year":_key_counts.shape[0],"duplicate_key_pairs":len(_dupes),"dupe_sample":list(_dupes.head(5).items()) if len(_dupes) else [],"streaming_long_rows":len(streaming_long)}})+"\n")
        _log.close()
        # #endregion

        # Join to canonical movies on normalized (title, year) to recover movie_id.
        # Allow many-to-many: same (title, year) can map to multiple movie_ids (remakes/dupes).
        streaming_joined = (
            streaming_long.merge(
                movies_join,
                on=["title_norm", "year"],
                how="inner",
            )
            .loc[:, ["movie_id", "provider"]]
            .drop_duplicates()
        )

        if streaming_joined.empty:
            print("Streaming join produced zero rows; not writing streaming.csv.")
        else:
            streaming_joined.to_csv(streaming_out_path, index=False)
            print(f"Saved {len(streaming_joined):,} streaming rows → {streaming_out_path}")
else:
    print(f"Streaming CSV not found at {streaming_raw_path}; skipping streaming.csv.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Tags: top 20 per movie, min count 2
# ─────────────────────────────────────────────────────────────────────────────
valid_movie_ids = set(movies["movie_id"])

tags_agg = (
    ml_tags[ml_tags["movieId"].isin(valid_movie_ids)]
    .groupby(["movieId", "tag"])
    .size()
    .reset_index(name="count")
)
tags_agg = tags_agg[tags_agg["count"] >= 2].copy()
tags_agg["tag"] = tags_agg["tag"].str.lower().str.strip()
tags_agg = (
    tags_agg.sort_values("count", ascending=False)
    .groupby("movieId")
    .head(20)
    .reset_index(drop=True)
)
tags_agg = tags_agg.rename(columns={"movieId": "movie_id"})
tags_agg.to_csv(f"{PROC}/tags.csv", index=False)
print(f"Saved {len(tags_agg):,} tag rows → {PROC}/tags.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Ratings: all ratings for movies in our canonical set (no sampling)
# Using all 25M makes collaborative filtering much more meaningful —
# popular movies get high co-rated counts but Jaccard corrects for that.
# ─────────────────────────────────────────────────────────────────────────────
ratings_filtered = ml_ratings[ml_ratings["movieId"].isin(valid_movie_ids)].copy()
ratings_filtered = ratings_filtered.rename(columns={"movieId": "movie_id", "userId": "user_id"})
ratings_filtered[["user_id", "movie_id", "rating"]].to_csv(
    f"{PROC}/ratings_sample.csv", index=False)
print(f"Saved {len(ratings_filtered):,} ratings → {PROC}/ratings_sample.csv")

print("\nDone! All processed files saved to data/processed/")
