"""
embedding.py – Build FAISS vector index from SHL catalog CSV

Run this script once (or whenever the catalog changes) to produce shl_index.faiss.

Design decisions:
- Embeddings are built from a rich text field combining name + test_type + description.
  Using only the name would miss semantic content; using full HTML content is noisy.
- normalize_embeddings=True ensures we can use inner product (dot product) as cosine
  similarity in FAISS — faster than L2 for retrieval.
- IndexFlatIP (inner product) is chosen over HNSW/IVF because the catalog is small
  (<2000 entries). For >100K entries, switch to IndexIVFFlat or HNSW.
- Batch encoding with progress reporting for transparency.
- Index metadata (CSV row order) is implicitly preserved: FAISS index row i
  corresponds to df.iloc[i].

Usage:
    python embedding.py
    python embedding.py --csv shl_catalog.csv --output shl_index.faiss
"""

import argparse
import logging
import os
import sys

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("shl_embedding")

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
DEFAULT_CSV = os.getenv("SHL_CSV_PATH", "shl_catalog.csv")
DEFAULT_INDEX = os.getenv("SHL_INDEX_PATH", "shl_index.faiss")
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_BATCH = 64


# ──────────────────────────────────────────────
# Text composition
# ──────────────────────────────────────────────

def _compose_text(row: pd.Series) -> str:
    """
    Compose a rich text representation for embedding.

    Field priority:
    1. name           — most discriminative, always present
    2. test_type      — adds categorical signal
    3. description    — adds semantic content (may be truncated in CSV)

    We concatenate with ". " separator so sentence transformers process
    each segment as a natural language sequence rather than a joined blob.
    """
    parts = []

    name = str(row.get("name", "")).strip()
    if name:
        parts.append(name)

    test_type = str(row.get("test_type", "")).strip()
    if test_type and test_type.lower() not in ("nan", "unknown", ""):
        parts.append(f"Test type: {test_type}")

    description = str(row.get("description", "")).strip()
    if description and description.lower() not in ("nan", ""):
        # Truncate to 512 chars — model max seq length is 256 tokens anyway
        parts.append(description[:512])

    return ". ".join(parts)


# ──────────────────────────────────────────────
# Index builder
# ──────────────────────────────────────────────

def build_index(
    csv_path: str = DEFAULT_CSV,
    index_path: str = DEFAULT_INDEX,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH,
) -> None:
    """
    Load catalog CSV, encode all entries, build FAISS index, save to disk.

    Args:
        csv_path:   Path to shl_catalog.csv
        index_path: Output path for FAISS index file
        model_name: sentence-transformers model name
        batch_size: Encoding batch size
    """
    # ── Load catalog ──────────────────────────────────────────
    logger.info(f"Loading catalog from: {csv_path}")
    if not os.path.exists(csv_path):
        logger.error(f"CSV not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    logger.info(f"Loaded {len(df)} catalog entries.")

    # ── Compose text for each entry ───────────────────────────
    texts = [_compose_text(row) for _, row in df.iterrows()]
    logger.info(f"Composed {len(texts)} text representations.")

    if not texts:
        logger.error("No texts to embed. Check your CSV.")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    embed_dim = model.get_sentence_embedding_dimension()
    logger.info(f"Embedding dimension: {embed_dim}")

    # ── Encode in batches ─────────────────────────────────────
    logger.info(f"Encoding {len(texts)} texts (batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # required for IndexFlatIP = cosine similarity
    ).astype(np.float32)

    logger.info(f"Embeddings shape: {embeddings.shape}")

    # ── Build FAISS index ─────────────────────────────────────
    logger.info("Building FAISS IndexFlatIP (inner product / cosine similarity)...")
    index = faiss.IndexFlatIP(embed_dim)
    index.add(embeddings)
    logger.info(f"FAISS index built with {index.ntotal} vectors.")

    # ── Save index ────────────────────────────────────────────
    logger.info(f"Saving FAISS index to: {index_path}")
    faiss.write_index(index, index_path)
    logger.info("Index saved successfully.")

    # ── Quick sanity check ────────────────────────────────────
    logger.info("Sanity check: querying 'Java developer assessment'...")
    test_vec = model.encode(
        ["Java developer assessment"],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    D, I = index.search(test_vec, 3)
    for rank, (dist, idx) in enumerate(zip(D[0], I[0]), 1):
        logger.info(f"  #{rank}: {df.iloc[idx]['name']} (score={dist:.4f})")

    logger.info("✅ Index build complete.")


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build FAISS embedding index from SHL catalog CSV."
    )
    parser.add_argument(
        "--csv", default=DEFAULT_CSV, help=f"Path to catalog CSV (default: {DEFAULT_CSV})"
    )
    parser.add_argument(
        "--output", default=DEFAULT_INDEX, help=f"Output FAISS index path (default: {DEFAULT_INDEX})"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Embedding model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH, help="Encoding batch size"
    )
    args = parser.parse_args()

    build_index(
        csv_path=args.csv,
        index_path=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
    )
