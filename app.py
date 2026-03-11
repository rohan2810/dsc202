"""
Movie Finder + Graph Recommender
Streamlit app — talks directly to Postgres, Neo4j, and Qdrant.

Run: streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import Optional
from urllib.parse import quote

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
)

TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w342"
PLACEHOLDER   = "https://via.placeholder.com/185x278/1a1a2e/e0e0e0?text=No+Poster"

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hero section ── */
.hero {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px;
    padding: 52px 48px 44px;
    margin-bottom: 28px;
    text-align: center;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 8px;
    letter-spacing: -0.5px;
}
.hero p {
    font-size: 1.1rem;
    color: #a0a8c8;
    margin: 0 0 32px;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.78rem;
    color: #c8d0e8;
    margin: 0 4px 8px;
}

/* ── Search bar ── */
.stTextInput > div > div > input {
    font-size: 1.05rem !important;
    padding: 14px 20px !important;
    border-radius: 12px !important;
    border: 2px solid #e0e4ef !important;
    background: #ffffff !important;
    color: #1a1a2e !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #6c63ff !important;
    box-shadow: 0 0 0 3px rgba(108,99,255,0.15) !important;
}

/* ── Movie card ── */
.movie-card {
    background: #ffffff;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    height: 100%;
    cursor: pointer;
    border: 1px solid #f0f2f8;
}
.movie-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 28px rgba(0,0,0,0.14);
}
.movie-poster {
    width: 100%;
    aspect-ratio: 2/3;
    object-fit: cover;
    display: block;
}
.movie-info {
    padding: 12px 14px 14px;
}
.movie-title {
    font-weight: 600;
    font-size: 0.9rem;
    color: #1a1a2e;
    line-height: 1.3;
    margin-bottom: 4px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.movie-year {
    font-size: 0.78rem;
    color: #7a7f9a;
    margin-bottom: 8px;
}
.movie-rating {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: #fff8e1;
    border: 1px solid #ffe082;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #f59e0b;
    margin-bottom: 8px;
}
.genre-pill {
    display: inline-block;
    background: #f0f0ff;
    color: #6c63ff;
    border-radius: 4px;
    padding: 1px 7px;
    font-size: 0.7rem;
    font-weight: 500;
    margin: 1px 2px 1px 0;
}
.card-streaming {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 4px;
}
.explain-badge {
    background: linear-gradient(135deg, #f0f0ff, #f8f5ff);
    border: 1px solid #ddd6fe;
    border-radius: 8px;
    padding: 8px 10px;
    margin-top: 8px;
    font-size: 0.73rem;
    color: #4c1d95;
    line-height: 1.5;
}
.explain-badge .signal {
    display: block;
    padding: 2px 0;
}

/* ── Section header ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 8px 0 20px;
}
.section-header h2 {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e0e4f0;
    margin: 0;
}
.result-count {
    background: #6c63ff;
    color: white;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.8rem;
    font-weight: 600;
}

/* ── Landing cards ── */
.tip-card {
    background: linear-gradient(135deg, #f8f9ff, #f0f0ff);
    border: 1px solid #e0e0f8;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    height: 100%;
}
.tip-icon { font-size: 2rem; margin-bottom: 8px; }
.tip-title { font-weight: 600; color: #1a1a2e; font-size: 0.95rem; margin-bottom: 6px; }
.tip-text  { color: #6b7280; font-size: 0.82rem; line-height: 1.5; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0f0c29 !important;
}
section[data-testid="stSidebar"] *:not(button):not(button *) {
    color: #e0e4f0 !important;
}
section[data-testid="stSidebar"] button {
    color: #1a1a2e !important;
    background: #e0e4f0 !important;
    border: 1px solid #c8d0e8 !important;
}
section[data-testid="stSidebar"] button:hover {
    background: #ffffff !important;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #ffffff !important;
    font-size: 1.1rem !important;
    line-height: 1.3 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1) !important;
}
.tag-chip {
    display: inline-block;
    background: rgba(108,99,255,0.25);
    border: 1px solid rgba(108,99,255,0.4);
    color: #b8b0ff !important;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    margin: 2px 2px 2px 0;
}
.streaming-link, .streaming-pill {
    color: #93c5fd !important;
    text-decoration: none;
    font-size: 0.85rem;
}
.streaming-link:hover {
    text-decoration: underline;
}
.streaming-pill {
    display: inline-block;
    background: rgba(59,130,246,0.2);
    border-radius: 4px;
    padding: 2px 8px;
}
.sidebar-rating {
    background: rgba(245,158,11,0.15);
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 8px 0;
    text-align: center;
}
.sidebar-rating .val { font-size: 1.6rem; font-weight: 700; color: #fbbf24 !important; }
.sidebar-rating .sub { font-size: 0.75rem; color: #9ca3af !important; }

/* ── Streamlit overrides ── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
}

/* ── Example query chips ── */
div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) .stButton > button {
    background: #f5f3ff !important;
    color: #5b4fcf !important;
    border: 1px solid #ddd6fe !important;
    border-radius: 20px !important;
    font-size: 0.78rem !important;
    padding: 4px 10px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    font-weight: 400 !important;
}
div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) .stButton > button:hover {
    background: #ede9fe !important;
    border-color: #a78bfa !important;
    color: #4c1d95 !important;
}
div[data-testid="stHorizontalBlock"] { gap: 14px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DB clients — cached so they load once per session
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model...")
def get_model():
    return SentenceTransformer("all-mpnet-base-v2")

@st.cache_resource(show_spinner="Connecting to Qdrant...")
def get_qdrant():
    return QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )

@st.cache_resource(show_spinner="Connecting to Neo4j...")
def get_neo4j():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=("neo4j", os.getenv("NEO4J_PASS", "moviepass")),
    )

@st.cache_resource(show_spinner="Connecting to Postgres...")
def get_pg():
    return psycopg2.connect(
        os.getenv("POSTGRES_DSN", "dbname=moviedb user=movie password=movie host=localhost")
    )

model  = get_model()
qdrant = get_qdrant()
neo4j  = get_neo4j()
pg     = get_pg()


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────
def _qdrant_vector_search(collection: str, query_vector: list, limit: int):
    """Run vector search; supports both qdrant-client search() and query_points() APIs."""
    if hasattr(qdrant, "search"):
        results = qdrant.search(
            collection_name=collection, query_vector=query_vector, limit=limit
        )
        return results if isinstance(results, list) else getattr(results, "points", results)
    return qdrant.query_points(collection, query=query_vector, limit=limit).points


def vector_search(query: str, limit: int = 20) -> list[dict]:
    """Search movies by natural language. Blends vector similarity with a
    quality signal so obscure low-rated films don't dominate results."""
    vec    = model.encode(query, normalize_embeddings=True).tolist()
    points = _qdrant_vector_search("movies", vec, limit * 5)

    candidates = []
    for h in points:
        avg = float(h.payload.get("bayesian_avg") or 0)
        # quality_boost: 0→0, 3.0→0.6, 3.5→0.7, 4.0→0.8, 5.0→1.0
        quality = min(avg / 5.0, 1.0) if avg > 0 else 0.0
        # Blend: 60% semantic relevance + 40% quality
        blended = 0.6 * h.score + 0.4 * quality
        candidates.append({
            "movie_id":     h.id,
            "title":        h.payload.get("title", ""),
            "year":         h.payload.get("year"),
            "genres":       h.payload.get("genres", ""),
            "poster_path":  h.payload.get("poster_path", ""),
            "bayesian_avg": avg,
            "score":        round(blended, 3),
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:limit]


def get_movie_detail(movie_id: int) -> dict:
    """Fetch movie metadata, tags, and streaming providers from Postgres."""
    with pg.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM movies WHERE movie_id = %s", (movie_id,))
        row = cur.fetchone()
        if not row:
            return {}
        movie = dict(row)
        cur.execute(
            "SELECT tag FROM movie_tags WHERE movie_id = %s ORDER BY count DESC LIMIT 12",
            (movie_id,),
        )
        movie["tags"] = [r["tag"] for r in cur.fetchall()]
        # Streaming availability (from movie_streaming if table exists)
        try:
            cur.execute(
                "SELECT provider FROM movie_streaming WHERE movie_id = %s ORDER BY provider",
                (movie_id,),
            )
            movie["streaming"] = [r["provider"] for r in cur.fetchall()]
        except psycopg2.ProgrammingError:
            movie["streaming"] = []
    return movie


def get_streaming_by_movie_ids(movie_ids: list[int]) -> dict[int, list[str]]:
    """Batch-fetch streaming providers for many movies. Returns dict movie_id -> [provider, ...]."""
    if not movie_ids:
        return {}
    out = {mid: [] for mid in movie_ids}
    try:
        with pg.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT movie_id, provider FROM movie_streaming WHERE movie_id = ANY(%s) ORDER BY movie_id, provider",
                (movie_ids,),
            )
            for row in cur.fetchall():
                out.setdefault(row["movie_id"], []).append(row["provider"])
    except psycopg2.ProgrammingError:
        pass
    return out


def _norm_provider(p: str) -> str:
    return str(p or "").strip().lower().replace("_", " ")


def graph_recommend(movie_id: int, limit: int = 20) -> list[dict]:
    scores: dict[int, dict] = {}

    with neo4j.session() as s:

        # ── Signal 1: Collaborative filtering (fans who loved this) ────────
        # Sample 5K fans (rated >= 4) and count co-loves. Skip full Jaccard
        # (counting ALL raters per candidate OOMs on 25M edges). The quality
        # gate (bayesian_avg >= 3.5) and high rating threshold already handle
        # popularity bias effectively.
        collab_rows = s.run("""
            MATCH (seed:Movie {movie_id: $mid})<-[r1:RATED]-(u:User)
            WHERE r1.rating >= 4.0
            WITH u ORDER BY u.user_id LIMIT 5000
            MATCH (u)-[r2:RATED]->(other:Movie)
            WHERE r2.rating >= 4.0
              AND other.movie_id <> $mid
              AND other.bayesian_avg >= 3.5
            WITH other, count(DISTINCT u) AS co_fans
            WHERE co_fans >= 5
            RETURN other.movie_id AS movie_id,
                   other.title AS title,
                   other.year AS year,
                   other.bayesian_avg AS bayesian_avg,
                   other.poster_path AS poster_path,
                   co_fans
            ORDER BY co_fans DESC LIMIT 60
        """, mid=movie_id).data()

        max_cofans = max((r["co_fans"] for r in collab_rows), default=1)
        for r in collab_rows:
            mid = r["movie_id"]
            scores[mid] = {
                **r,
                "composite":   (r["co_fans"] / max_cofans) * 0.25,
                "explanation": f"⭐ {r['co_fans']} fans also loved this",
            }

        # ── Signal 2: Shared tags (weighted by tag frequency) ─────────────
        # Use HAS_TAG.count to weight overlap: sum(min(seed_count, other_count)) across shared tags.
        tag_rows = s.run("""
            MATCH (m:Movie {movie_id: $mid})-[r1:HAS_TAG]->(t:Tag)<-[r2:HAS_TAG]-(other:Movie)
            WHERE other.movie_id <> $mid AND other.bayesian_avg >= 3.0
            WITH other,
                 collect(t.name) AS shared_tags,
                 sum(CASE WHEN r1.count < r2.count THEN r1.count ELSE r2.count END) AS weighted_overlap
            ORDER BY weighted_overlap DESC LIMIT 80
            RETURN other.movie_id AS movie_id,
                   other.title AS title,
                   other.year AS year,
                   other.bayesian_avg AS bayesian_avg,
                   other.poster_path AS poster_path,
                   shared_tags, weighted_overlap
        """, mid=movie_id).data()

        max_weighted = max((r["weighted_overlap"] for r in tag_rows), default=1)
        for r in tag_rows:
            mid   = r["movie_id"]
            boost = (r["weighted_overlap"] / max_weighted) * 0.35
            tag_label = ", ".join(r["shared_tags"][:5])
            if mid not in scores:
                scores[mid] = {**r, "composite": 0.0,
                               "explanation": f"🏷️ Shared themes: {tag_label}"}
            else:
                scores[mid]["explanation"] += f" · 🏷️ Shared themes: {tag_label}"
            scores[mid]["composite"] += boost

        # ── Signal 3: Genre match — quality-gated ─────────────────────────
        for r in s.run("""
            MATCH (m:Movie {movie_id: $mid})-[:HAS_GENRE]->(g:Genre)<-[:HAS_GENRE]-(other:Movie)
            WHERE other.movie_id <> $mid AND other.bayesian_avg >= 3.5
            WITH other, collect(g.name) AS shared_genres, count(DISTINCT g) AS genre_count
            ORDER BY genre_count DESC, other.bayesian_avg DESC LIMIT 60
            RETURN other.movie_id AS movie_id,
                   other.title AS title,
                   other.year AS year,
                   other.bayesian_avg AS bayesian_avg,
                   other.poster_path AS poster_path,
                   shared_genres, genre_count
        """, mid=movie_id).data():
            mid   = r["movie_id"]
            boost = min(r["genre_count"] / 3.0, 1.0) * 0.15
            glabel = ", ".join(r["shared_genres"][:3])
            if mid not in scores:
                scores[mid] = {**r, "composite": 0.0,
                               "explanation": f"🎬 Same genres: {glabel}"}
            else:
                scores[mid]["explanation"] += f" · 🎬 {glabel}"
            scores[mid]["composite"] += boost

    # ── Signal 4: Vector similarity from Qdrant ───────────────────────────
    try:
        seed_pts = qdrant.retrieve("movies", ids=[movie_id], with_vectors=True)
        if seed_pts and seed_pts[0].vector:
            vec   = seed_pts[0].vector
            hits = _qdrant_vector_search("movies", vec, limit + 5)
            max_vscore = max(
                (h.score for h in hits if h.id != movie_id), default=1.0
            )
            for h in hits:
                if h.id == movie_id:
                    continue
                mid   = h.id
                boost = (h.score / max_vscore) * 0.25
                if mid not in scores:
                    scores[mid] = {
                        "movie_id":    mid,
                        "title":       h.payload.get("title", ""),
                        "year":        h.payload.get("year"),
                        "bayesian_avg": h.payload.get("bayesian_avg", 0.0),
                        "poster_path": h.payload.get("poster_path", ""),
                        "composite":   0.0,
                        "explanation": "🔍 Similar plot & style",
                    }
                scores[mid]["composite"] += boost
    except Exception:
        pass

    ranked = sorted(scores.values(), key=lambda x: x["composite"], reverse=True)
    top = ranked[:limit]

    # Backfill genres for results that came from Neo4j (which doesn't store genres on node)
    missing = [m["movie_id"] for m in top if not m.get("genres")]
    if missing:
        try:
            hits = qdrant.retrieve("movies", ids=missing, with_payload=True)
            genre_map = {h.id: h.payload.get("genres", "") for h in hits}
            for m in top:
                if not m.get("genres") and m["movie_id"] in genre_map:
                    m["genres"] = genre_map[m["movie_id"]]
        except Exception:
            pass

    return top


def recommend_streaming_filtered(movie_id: int, provider: str, limit: int = 20) -> list[dict]:
    """
    Run the normal recommendation engine, then filter results to movies that are
    available on the given streaming provider (inclusive: may also be on others).
    """
    p_norm = _norm_provider(provider)
    overfetch = max(limit * 5, 80)
    candidates = graph_recommend(movie_id, limit=overfetch)
    if not candidates:
        return []

    ids = [m["movie_id"] for m in candidates]
    streaming_map = get_streaming_by_movie_ids(ids)

    filtered = []
    for m in candidates:
        provs = [_norm_provider(x) for x in (streaming_map.get(m["movie_id"], []) or [])]
        if p_norm in provs:
            filtered.append(m)

    return filtered[:limit]


def poster_url(path: Optional[str]) -> str:
    """Return a full TMDB poster URL or a placeholder when no valid path is provided."""
    if path and str(path) not in ("nan", "None", ""):
        return TMDB_IMG_BASE + str(path)
    return PLACEHOLDER


# ─────────────────────────────────────────────────────────────────────────────
# UI components
# ─────────────────────────────────────────────────────────────────────────────
def render_card(col, movie: dict, mode: str = "search"):
    """Render a styled movie card inside a column."""
    with col:
        title   = movie.get("title", "Unknown")
        year    = movie.get("year", "")
        genres  = str(movie.get("genres", "") or "").replace("|", ",").split(",")
        genres  = [g.strip() for g in genres if g.strip()][:3]
        avg     = float(movie.get("bayesian_avg") or 0)
        explain = movie.get("explanation", "")
        img     = poster_url(movie.get("poster_path"))

        genre_pills = "".join(f'<span class="genre-pill">{g}</span>' for g in genres)
        rating_html = f'<div class="movie-rating">★ {avg:.1f}</div>' if avg > 0 else ""
        streaming = movie.get("streaming") or []
        streaming_html = ""
        if streaming:
            labels = [p.replace("_", " ").title() for p in streaming]
            streaming_html = f'<div class="card-streaming">Watch: {", ".join(labels)}</div>'
        if mode == "recommend" and explain:
            signals = explain.split(" · ")
            lines = "".join(f'<span class="signal">{s}</span>' for s in signals)
            explain_html = f'<div class="explain-badge"><b>Why this?</b>{lines}</div>'
        else:
            explain_html = ""

        card_html = (
            f'<div class="movie-card">'
            f'<img class="movie-poster" src="{img}" onerror="this.src=\'{PLACEHOLDER}\'" />'
            f'<div class="movie-info">'
            f'<div class="movie-title">{title}</div>'
            f'<div class="movie-year">{year}</div>'
            f'{rating_html}'
            f'<div>{genre_pills}</div>'
            f'{streaming_html}'
            f'{explain_html}'
            f'</div></div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("Details / Recommend →", key=f"card_{movie['movie_id']}_{mode}",
                     use_container_width=True):
            st.session_state["selected_id"] = movie["movie_id"]
            st.rerun()


def render_grid(movies: list[dict], mode: str = "search"):
    """Render a grid of movie cards; enriches each movie with streaming providers when available."""
    ids = [m["movie_id"] for m in movies]
    streaming_map = get_streaming_by_movie_ids(ids)
    for m in movies:
        m["streaming"] = streaming_map.get(m["movie_id"], [])
    cols = st.columns(5, gap="medium")
    for i, movie in enumerate(movies):
        render_card(cols[i % 5], movie, mode=mode)


def render_detail_sidebar(movie_id: int):
    """Show movie details + action buttons in the sidebar."""
    detail = get_movie_detail(movie_id)
    if not detail:
        st.sidebar.error("Movie not found in database.")
        return

    with st.sidebar:
        st.image(poster_url(detail.get("poster_path")))

        st.markdown(
            f'<h3 style="margin-top:12px">{detail.get("title","")}'
            f' <span style="font-weight:300;color:#8892b0">({detail.get("year","")})</span></h3>',
            unsafe_allow_html=True,
        )

        # Rating block
        rating = detail.get("rating_mean")
        count  = detail.get("rating_count")
        if rating and count:
            st.markdown(
                f'<div class="sidebar-rating">'
                f'<div class="val">★ {float(rating):.2f}</div>'
                f'<div class="sub">{int(count):,} ratings</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if detail.get("director"):
            st.markdown(f"**Director:** {detail['director']}")
        if detail.get("top_cast"):
            st.markdown(f"**Cast:** {detail['top_cast']}")

        # Genre pills in sidebar
        genres = str(detail.get("genres","") or "").replace("|", ",").split(",")
        genres = [g.strip() for g in genres if g.strip()]
        if genres:
            pills = " ".join(
                f'<span style="display:inline-block;background:rgba(108,99,255,0.2);'
                f'color:#b8b0ff;border-radius:4px;padding:2px 8px;font-size:0.72rem;'
                f'margin:2px">{g}</span>' for g in genres
            )
            st.markdown(f"**Genres:** {pills}", unsafe_allow_html=True)

        # Tag chips
        if detail.get("tags"):
            chips = " ".join(
                f'<span class="tag-chip">{t}</span>' for t in detail["tags"]
            )
            st.markdown(f"**Tags:** {chips}", unsafe_allow_html=True)

        # Streaming: link to each provider's search when data is available
        if detail.get("streaming"):
            title_enc = quote(str(detail.get("title", "")))
            # Map stored provider names to display labels and search URLs
            provider_urls = {
                "netflix": ("Netflix", "https://www.netflix.com/search?q="),
                "hulu": ("Hulu", "https://www.hulu.com/search?q="),
                "prime video": ("Prime Video", "https://www.primevideo.com/search/ref=atv_nb_sr?phrase="),
                "disney+": ("Disney+", "https://www.disneyplus.com/search?q="),
            }
            links = []
            for p in detail["streaming"]:
                p_norm = _norm_provider(p)
                label, base = provider_urls.get(p_norm, (p.replace("_", " ").title(), None))
                if base:
                    links.append(f'<a href="{base}{title_enc}" target="_blank" rel="noopener" class="streaming-link">{label}</a>')
                else:
                    links.append(f'<span class="streaming-pill">{label}</span>')
            st.markdown(
                "**Where to watch:** " + " · ".join(links),
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            for p in detail["streaming"]:
                p_norm = _norm_provider(p)
                label, _base = provider_urls.get(p_norm, (p.replace("_", " ").title(), None))
                if st.button(
                    f"See what else is streaming on {label}",
                    key=f"stream_rec_{movie_id}_{p_norm}",
                    use_container_width=True,
                ):
                    st.session_state["rec_id"] = movie_id
                    st.session_state["rec_title"] = detail.get("title", "")
                    st.session_state["rec_provider"] = p_norm
                    st.session_state.pop("selected_id", None)
                    st.rerun()

        if detail.get("overview"):
            st.markdown("---")
            st.markdown(f'<p style="font-size:0.85rem;line-height:1.6;color:#c8d0e8">'
                        f'{detail["overview"]}</p>', unsafe_allow_html=True)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Find Similar", use_container_width=True):
                st.session_state["rec_id"]    = movie_id
                st.session_state["rec_title"] = detail.get("title", "")
                st.session_state.pop("selected_id", None)
                st.rerun()
        with col2:
            if st.button("✕ Close", use_container_width=True):
                st.session_state.pop("selected_id", None)
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main app layout
# ─────────────────────────────────────────────────────────────────────────────

# Show detail sidebar if a movie was clicked
if "selected_id" in st.session_state:
    render_detail_sidebar(st.session_state["selected_id"])

# ── Hero / search bar ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎬 CineMatch</h1>
    <p>Describe any movie in plain English — we'll find it and recommend what to watch next</p>
    <span class="hero-badge">🧠 Semantic Search</span>
    <span class="hero-badge">🕸️ Graph Recommendations</span>
    <span class="hero-badge">📊 25M Ratings</span>
    <span class="hero-badge">🎯 61K Movies</span>
</div>
""", unsafe_allow_html=True)

# Handle example chip clicks — set widget default BEFORE it renders
EXAMPLE_QUERIES = [
    "mind-bending sci-fi with time travel",
    "feel-good comedy about friendship",
    "slow-burn psychological thriller",
    "survival film against nature",
    "animated movie about loss and grief",
    "heist with a clever twist ending",
]

if "chip_query" in st.session_state:
    st.session_state["q"] = st.session_state.pop("chip_query")

query = st.text_input(
    "Search",
    key="q",
    placeholder='e.g. "mind-bending sci-fi with time travel but not too scary"',
    label_visibility="collapsed",
)

# ── Example query chips ────────────────────────────────────────────────────
st.markdown("<div style='margin: 10px 0 4px; font-size:0.78rem; color:#9ca3af; font-weight:500'>Try a search:</div>", unsafe_allow_html=True)
chip_cols = st.columns(len(EXAMPLE_QUERIES))
for col, example in zip(chip_cols, EXAMPLE_QUERIES):
    with col:
        if st.button(example, key=f"ex_{example}", use_container_width=True):
            st.session_state["chip_query"] = example
            st.rerun()

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── Recommendation mode ────────────────────────────────────────────────────
if "rec_id" in st.session_state:
    rec_id    = st.session_state["rec_id"]
    rec_title = st.session_state["rec_title"]
    rec_provider = st.session_state.get("rec_provider")

    col_a, col_b = st.columns([7, 1])
    with col_a:
        if rec_provider:
            provider_label = str(rec_provider).replace("_", " ").title()
            header = f'Because you liked <em>{rec_title}</em> (and want <em>{provider_label}</em>)…'
        else:
            header = f'Because you liked <em>{rec_title}</em>…'
        st.markdown(
            f'<div class="section-header">'
            f'<h2>{header}</h2>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_b:
        if st.button("← Back"):
            st.session_state.pop("rec_id", None)
            st.session_state.pop("rec_title", None)
            st.session_state.pop("rec_provider", None)
            st.rerun()

    with st.spinner("Traversing graph..."):
        if rec_provider:
            recs = recommend_streaming_filtered(rec_id, str(rec_provider))
        else:
            recs = graph_recommend(rec_id)

    if recs:
        render_grid(recs, mode="recommend")
    else:
        st.info("No recommendations found. The graph may not have enough overlap for this movie.")

# ── Search mode ────────────────────────────────────────────────────────────
elif query:
    with st.spinner("Searching..."):
        results = vector_search(query)

    st.markdown(
        f'<div class="section-header">'
        f'<h2>Results for <em>"{query}"</em></h2>'
        f'<span class="result-count">{len(results)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if results:
        render_grid(results, mode="search")
    else:
        st.info("No results found. Try a different query.")

# ── Landing state ──────────────────────────────────────────────────────────
else:
    c1, c2, c3 = st.columns(3, gap="large")
    tips = [
        ("🔍", "Natural Language Search",
         'Try "a slow-burn thriller set in Japan" or "feel-good comedy about food"'),
        ("🕸️", "Graph Recommendations",
         "Click any movie poster, then hit Find Similar to traverse 25M ratings across Neo4j"),
        ("🎯", "Multi-Signal Ranking",
         "Results blend collaborative filtering, shared tags, genre overlap, and semantic similarity"),
    ]
    for col, (icon, title, text) in zip([c1, c2, c3], tips):
        with col:
            st.markdown(
                f'<div class="tip-card">'
                f'<div class="tip-icon">{icon}</div>'
                f'<div class="tip-title">{title}</div>'
                f'<div class="tip-text">{text}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
