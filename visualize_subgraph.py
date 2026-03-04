"""
Build a small subgraph from Neo4j (3–4 movies, ~10 users, genres and tags)
and render it as a labeled figure with legend. Uses small Cypher queries to
avoid Neo4j transaction memory limits.

Run from repo root: python visualize_subgraph.py

Output: subgraph.png (and optionally subgraph.html if pyvis is installed).
"""

import os
import sys
from collections import defaultdict

from dotenv import load_dotenv
from neo4j import GraphDatabase
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_PASS = os.getenv("NEO4J_PASS", "moviepass")
NUM_MOVIES = 4
NUM_USERS = 10
MAX_GENRES = 14
MAX_TAGS = 16
OUTPUT_PNG = "subgraph.png"


def run_query(driver, query, **params):
    with driver.session() as session:
        result = session.run(query, **params)
        return [dict(record) for record in result]


def fetch_subgraph_data(driver):
    """Run several small queries to get movies, genres, tags, and ratings."""
    # 1) Get a few movie IDs that share a genre and have at least one tag
    movies_q = """
        MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre), (m)-[:HAS_TAG]->()
        WHERE m.bayesian_avg >= 3.5
        WITH g, collect(DISTINCT m.movie_id) AS ids
        WHERE size(ids) >= 4
        RETURN ids[0..4] AS movie_ids
        LIMIT 1
    """
    rows = run_query(driver, movies_q)
    if not rows or not rows[0].get("movie_ids"):
        # Fallback: any 4 movies with genre and tag
        movies_q2 = """
            MATCH (m:Movie)-[:HAS_GENRE]->(), (m)-[:HAS_TAG]->()
            WHERE m.bayesian_avg >= 3.5
            RETURN m.movie_id AS movie_id
            LIMIT 4
        """
        rows2 = run_query(driver, movies_q2)
        movie_ids = [r["movie_id"] for r in rows2]
    else:
        movie_ids = rows[0]["movie_ids"]

    if len(movie_ids) < 2:
        print("Need at least 2 movies with genre and tag. Exiting.")
        sys.exit(1)

    # 2) Get movie titles (one small query)
    titles_q = """
        UNWIND $movie_ids AS mid
        MATCH (m:Movie {movie_id: mid})
        RETURN m.movie_id AS movie_id, m.title AS title
    """
    titles = {r["movie_id"]: (r["title"] or "?")[:30] for r in run_query(driver, titles_q, movie_ids=movie_ids)}

    # 3) Genre edges (small)
    genre_q = """
        UNWIND $movie_ids AS mid
        MATCH (m:Movie {movie_id: mid})-[r:HAS_GENRE]->(g:Genre)
        RETURN mid AS movie_id, g.name AS genre_name
        LIMIT $max
    """
    genre_edges = run_query(driver, genre_q, movie_ids=movie_ids, max=MAX_GENRES)

    # 4) Tag edges (small)
    tag_q = """
        UNWIND $movie_ids AS mid
        MATCH (m:Movie {movie_id: mid})-[r:HAS_TAG]->(t:Tag)
        RETURN mid AS movie_id, t.name AS tag_name
        LIMIT $max
    """
    tag_edges = run_query(driver, tag_q, movie_ids=movie_ids, max=MAX_TAGS)

    # 5) Users who rated at least 2 of our movies (overlap), then their ratings
    overlapping_users_q = """
        UNWIND $movie_ids AS mid
        MATCH (u:User)-[:RATED]->(m:Movie {movie_id: mid})
        WITH u.user_id AS user_id, count(DISTINCT m) AS movies_rated
        WHERE movies_rated >= 2
        RETURN user_id
        LIMIT $max_users
    """
    overlapping = run_query(driver, overlapping_users_q, movie_ids=movie_ids, max_users=NUM_USERS)
    user_ids = [r["user_id"] for r in overlapping]

    if not user_ids:
        # Fallback: any users who rated any of our movies (no overlap guarantee)
        rating_q = """
            UNWIND $movie_ids AS mid
            MATCH (u:User)-[r:RATED]->(m:Movie {movie_id: mid})
            RETURN u.user_id AS user_id, mid AS movie_id, r.rating AS rating
            LIMIT $max
        """
        rating_edges = run_query(driver, rating_q, movie_ids=movie_ids, max=NUM_USERS * 2)
    else:
        # All (user, movie, rating) for overlapping users and our movies
        rating_q = """
            UNWIND $user_ids AS uid
            UNWIND $movie_ids AS mid
            MATCH (u:User {user_id: uid})-[r:RATED]->(m:Movie {movie_id: mid})
            RETURN u.user_id AS user_id, mid AS movie_id, r.rating AS rating
        """
        rating_edges = run_query(driver, rating_q, user_ids=user_ids, movie_ids=movie_ids)

    return {
        "movie_ids": movie_ids,
        "titles": titles,
        "genre_edges": genre_edges,
        "tag_edges": tag_edges,
        "rating_edges": rating_edges,
    }


def build_networkx_graph(data):
    G = nx.DiGraph()
    node_type = {}
    edge_type = {}

    def movie_node(mid):
        return ("Movie", mid)

    def genre_node(name):
        return ("Genre", name)

    def tag_node(name):
        return ("Tag", name)

    def user_node(uid):
        return ("User", uid)

    for mid in data["movie_ids"]:
        n = movie_node(mid)
        G.add_node(n)
        node_type[n] = "Movie"

    for r in data["genre_edges"]:
        mid, gname = r["movie_id"], r["genre_name"]
        m, g = movie_node(mid), genre_node(gname)
        G.add_node(g)
        node_type[g] = "Genre"
        G.add_edge(m, g)
        edge_type[(m, g)] = "HAS_GENRE"

    for r in data["tag_edges"]:
        mid, tname = r["movie_id"], r["tag_name"]
        m, t = movie_node(mid), tag_node(tname)
        G.add_node(t)
        node_type[t] = "Tag"
        G.add_edge(m, t)
        edge_type[(m, t)] = "HAS_TAG"

    seen_users = set()
    for r in data["rating_edges"]:
        uid, mid, rating = r["user_id"], r["movie_id"], r["rating"]
        if len(seen_users) >= NUM_USERS and uid not in seen_users:
            continue
        seen_users.add(uid)
        u, m = user_node(uid), movie_node(mid)
        G.add_node(u)
        node_type[u] = "User"
        G.add_edge(u, m, rating=rating)
        edge_type[(u, m)] = "RATED"

    return G, node_type, edge_type, data


def node_label(key, data):
    kind, val = key
    if kind == "Movie":
        return data["titles"].get(val, f"Movie {val}")
    if kind == "Genre":
        return str(val)[:12]
    if kind == "Tag":
        return str(val)[:12]
    if kind == "User":
        return f"User {val}"
    return str(val)[:12]


def draw_graph(G, node_type, edge_type, data):
    # Layout
    pos = nx.spring_layout(G, k=1.2, iterations=50, seed=42)

    # Colors
    NODE_COLOR = {
        "Movie": "#e74c3c",
        "Genre": "#3498db",
        "Tag": "#2ecc71",
        "User": "#9b59b6",
    }
    EDGE_COLOR = {
        "HAS_GENRE": "#3498db",
        "HAS_TAG": "#2ecc71",
        "RATED": "#9b59b6",
    }

    node_colors = [NODE_COLOR.get(node_type[n], "#95a5a6") for n in G.nodes()]
    edge_colors = [EDGE_COLOR.get(edge_type.get((u, v)), "#bdc3c7") for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=(14, 10))
    labels = {n: node_label(n, data) for n in G.nodes()}

    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=800,
        alpha=0.9,
        ax=ax,
    )
    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        arrows=True,
        arrowsize=14,
        width=1.5,
        connectionstyle="arc3,rad=0.05",
        ax=ax,
    )
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=8,
        font_weight="bold",
        ax=ax,
    )

    # Legend: node types
    node_leg = [
        mpatches.Patch(color=NODE_COLOR["Movie"], label="Movie"),
        mpatches.Patch(color=NODE_COLOR["Genre"], label="Genre"),
        mpatches.Patch(color=NODE_COLOR["Tag"], label="Tag"),
        mpatches.Patch(color=NODE_COLOR["User"], label="User"),
    ]
    edge_leg = [
        mpatches.Patch(color=EDGE_COLOR["HAS_GENRE"], label="HAS_GENRE"),
        mpatches.Patch(color=EDGE_COLOR["HAS_TAG"], label="HAS_TAG"),
        mpatches.Patch(color=EDGE_COLOR["RATED"], label="RATED"),
    ]
    ax.legend(
        handles=node_leg + edge_leg,
        loc="upper left",
        fontsize=9,
        title="Nodes & edges",
    )
    ax.set_title("Neo4j subgraph: Movies, Genres, Tags, Users (RATED)", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {OUTPUT_PNG}")


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=("neo4j", NEO4J_PASS))
    try:
        data = fetch_subgraph_data(driver)
        G, node_type, edge_type, data = build_networkx_graph(data)
        draw_graph(G, node_type, edge_type, data)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
