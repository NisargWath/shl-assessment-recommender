"""
retriever.py – FAISS-based semantic retrieval with reranking

Design decisions:
- Uses sentence-transformers "all-MiniLM-L6-v2" for query encoding
  (same model used during index build → consistent embedding space)
- Retrieves top-K*2 candidates from FAISS, then reranks by:
    1. Cosine similarity score from FAISS (primary)
    2. Keyword overlap boost (secondary) – improves Recall@10 on
       short technical queries like "Java developer"
- Deduplication by name prevents returning the same assessment twice
  when catalog has near-duplicate entries
- Query rewriting: expands short queries with domain context before
  embedding (e.g. "Java" → "Java programming skills assessment test")
- Fallback: if FAISS returns fewer than requested, returns all available
"""

import logging
import os
import re

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("shl_retriever")

# ──────────────────────────────────────────────
# Paths (relative to project root or configurable via env)
# ──────────────────────────────────────────────
_CSV_PATH = os.getenv("SHL_CSV_PATH", "shl_catalog.csv")
_INDEX_PATH = os.getenv("SHL_INDEX_PATH", "shl_index.faiss")
_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ──────────────────────────────────────────────
# Lazy-loaded singletons (load once on first call)
# ──────────────────────────────────────────────
_model: SentenceTransformer | None = None
_index: faiss.Index | None = None
_df: pd.DataFrame | None = None


def _load_resources():
    """Load embedding model, FAISS index, and catalog dataframe once."""
    global _model, _index, _df

    if _model is None:
        logger.info(f"Loading embedding model: {_MODEL_NAME}")
        _model = SentenceTransformer(_MODEL_NAME)

    if _index is None:
        logger.info(f"Loading FAISS index from: {_INDEX_PATH}")
        _index = faiss.read_index(_INDEX_PATH)
        logger.info(f"FAISS index loaded with {_index.ntotal} vectors.")

    if _df is None:
        logger.info(f"Loading catalog CSV from: {_CSV_PATH}")
        _df = pd.read_csv(_CSV_PATH)
        # Normalise column names to lowercase stripped
        _df.columns = [c.strip().lower() for c in _df.columns]
        logger.info(f"Catalog loaded: {len(_df)} entries.")


# ──────────────────────────────────────────────
# Query rewriting
# ──────────────────────────────────────────────

# Domain expansion hints: very short technical terms that benefit from
# expansion into a more semantically rich query string.
_DOMAIN_HINTS = {
    r"\bjava\b": "Java programming language skills test assessment",
    r"\bpython\b": "Python programming language coding assessment",
    r"\bsql\b": "SQL database query skills assessment",
    r"\bc\+\+\b": "C++ programming language skills assessment",
    r"\b\.net\b": ".NET framework programming assessment",
    r"\bjavascript\b": "JavaScript programming skills assessment",
    r"\breact\b": "React JavaScript frontend framework assessment",
    r"\bmanager\b": "management leadership skills assessment",
    r"\bsales\b": "sales skills personality assessment",
    r"\bcustomer service\b": "customer service communication skills assessment",
    r"\bcall center\b": "call center agent customer support assessment",
    r"\baccounting\b": "accounting bookkeeping financial skills assessment",
    r"\bbanking\b": "banking financial services assessment",
    r"\bretail\b": "retail cashier sales assessment",
    r"\bcognitive\b": "cognitive ability reasoning numerical verbal assessment",
    r"\bpersonality\b": "personality behavioral traits assessment",
    r"\bentry.?level\b": "entry level graduate apprentice job assessment",
    r"\bsenior\b": "senior experienced professional leadership assessment",
}


def _rewrite_query(query: str) -> str:
    """
    Expand short/sparse queries with domain hints.

    Returns an enriched query string that will embed more meaningfully.
    The original query is always preserved as prefix.
    """
    query_lower = query.lower()
    expansions = []
    for pattern, hint in _DOMAIN_HINTS.items():
        if re.search(pattern, query_lower, re.IGNORECASE):
            expansions.append(hint)

    if expansions:
        enriched = query + " | " + " | ".join(set(expansions))
        logger.debug(f"Query rewritten: '{query}' → '{enriched[:120]}'")
        return enriched
    return query


# ──────────────────────────────────────────────
# Keyword overlap scoring (secondary ranker)
# ──────────────────────────────────────────────

def _keyword_overlap_score(query: str, doc_text: str) -> float:
    """
    Soft keyword overlap score in [0.0, 1.0].

    Tokenises both strings to lowercase words and computes
    Jaccard-like overlap. Used to boost docs that share
    exact technical terms with the query (e.g. "Java").
    """
    query_tokens = set(re.findall(r"\w+", query.lower()))
    doc_tokens = set(re.findall(r"\w+", doc_text.lower()))
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & doc_tokens)
    return overlap / len(query_tokens)


# ──────────────────────────────────────────────
# Main retrieval function
# ──────────────────────────────────────────────

def retrieve_assessments(query: str, k: int = 10) -> list[dict]:
    """
    Retrieve the top-k most relevant SHL assessments for a query.

    Steps:
    1. Rewrite query for better semantic coverage.
    2. Embed with sentence-transformer.
    3. FAISS ANN search for top k*2 candidates (over-fetch then rerank).
    4. Score each candidate: cosine_score + 0.3 * keyword_overlap.
    5. Sort by combined score descending.
    6. Deduplicate by assessment name.
    7. Return top-k as list of dicts.

    Returns:
        List of dicts with keys: name, url, test_type, description, score
    """
    _load_resources()

    if not query or not query.strip():
        logger.warning("Empty query passed to retrieve_assessments.")
        return []

    # ── Step 1: Query rewriting ────────────────────────────────
    enriched_query = _rewrite_query(query)

    # ── Step 2: Embed ──────────────────────────────────────────
    query_vec = _model.encode(
        [enriched_query],
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine similarity via dot product
    ).astype(np.float32)

    # ── Step 3: FAISS search (over-fetch) ─────────────────────
    # Fetch min(k*2, index_size) candidates
    fetch_k = min(k * 2, _index.ntotal)
    distances, indices = _index.search(query_vec, fetch_k)

    candidates = []
    seen_names = set()

    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(_df):
            continue  # FAISS may return -1 for empty slots

        row = _df.iloc[idx]
        name = str(row.get("name", "")).strip()

        if not name or name.lower() in seen_names:
            continue  # deduplicate

        # Description field for keyword overlap (may be truncated in CSV)
        desc = str(row.get("description", ""))

        # Combined score: FAISS inner product (cosine) + keyword boost
        cosine_score = float(dist)  # already normalised
        keyword_score = _keyword_overlap_score(query, name + " " + desc)
        combined_score = cosine_score + 0.3 * keyword_score

        candidates.append(
            {
                "name": name,
                "url": str(row.get("url", "")),
                "test_type": str(row.get("test_type", "Unknown")),
                "description": desc[:400],  # truncate for prompt length
                "score": combined_score,
            }
        )
        seen_names.add(name.lower())

    # ── Step 4: Sort by combined score ────────────────────────
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # ── Step 5: Return top-k ───────────────────────────────────
    top_k = candidates[:k]
    logger.debug(f"Top-{k} retrieved: {[c['name'] for c in top_k]}")
    return top_k


def get_assessment_by_name(name: str) -> dict | None:
    """
    Exact-match lookup by assessment name (case-insensitive).
    Used for comparison queries where the user names a specific assessment.
    """
    _load_resources()
    name_lower = name.strip().lower()
    for _, row in _df.iterrows():
        if str(row.get("name", "")).strip().lower() == name_lower:
            return {
                "name": str(row["name"]),
                "url": str(row.get("url", "")),
                "test_type": str(row.get("test_type", "Unknown")),
                "description": str(row.get("description", "")),
            }
    return None


def get_all_assessment_names() -> list[str]:
    """Return all assessment names in the catalog (for validation)."""
    _load_resources()
    return [str(n) for n in _df["name"].dropna().tolist()]
