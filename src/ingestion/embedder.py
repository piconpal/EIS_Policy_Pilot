"""
embedder.py — Step 4
Embeds text chunks using sentence-transformers (all-MiniLM-L6-v2) and
persists them into a ChromaDB collection at vectorstore_path.

Pipeline:
  1. Load all processed chunk JSONs from data/processed/
  2. Load embedding model locally (no API calls)
  3. Batch-embed chunk texts
  4. Upsert into ChromaDB with full metadata
  5. Write an ingestion log to logs/ingestion/
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# ChromaDB collection name — single collection for the whole knowledge base
COLLECTION_NAME = "enterprise_rag"

# Embedding batch size — keeps memory usage predictable
BATCH_SIZE = None  # resolved from config at runtime


def _load_config() -> dict:
    """Load config.yaml from project root."""
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Chunk loader ───────────────────────────────────────────────────────────────

def load_processed_chunks(processed_data_path: str) -> list[dict]:
    """
    Load all *_chunks.json files from data/processed/.

    Returns:
        Flat list of all chunk dicts across every processed PDF.

    Raises:
        FileNotFoundError: If processed_data_path does not exist.
        ValueError: If no chunk files are found.
    """
    project_root   = Path(__file__).resolve().parents[2]
    processed_dir  = (project_root / processed_data_path).resolve()

    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Processed data directory not found: {processed_dir}\n"
            "Run pdf_loader + chunker first."
        )

    chunk_files = sorted(processed_dir.glob("*_chunks.json"))
    if not chunk_files:
        raise ValueError(
            f"No chunk files found in {processed_dir}.\n"
            "Run the chunker (Step 3) to generate them."
        )

    all_chunks: list[dict] = []
    for path in chunk_files:
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        logger.info(f"  Loaded {len(chunks):>5} chunks from {path.name}")
        all_chunks.extend(chunks)

    logger.info(f"Total chunks loaded: {len(all_chunks)}")
    return all_chunks


# ── ChromaDB client ────────────────────────────────────────────────────────────

def _get_chroma_collection(vectorstore_path: str) -> chromadb.Collection:
    """
    Create (or open) a persistent ChromaDB collection.

    Args:
        vectorstore_path: Relative path to the vectorstore directory.

    Returns:
        ChromaDB Collection object.
    """
    project_root   = Path(__file__).resolve().parents[2]
    persist_dir    = (project_root / vectorstore_path).resolve()
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )

    # get_or_create — safe to call on every run
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine similarity for retrieval
    )
    logger.info(
        f"ChromaDB collection '{COLLECTION_NAME}' ready | "
        f"existing docs: {collection.count()}"
    )
    return collection


# ── Ingestion log ──────────────────────────────────────────────────────────────

def _write_ingestion_log(
    log_dir: str,
    total_chunks: int,
    new_chunks: int,
    skipped_chunks: int,
    source_files: list[str],
    duration_s: float,
    model_name: str,
) -> None:
    """
    Write a JSON ingestion summary to logs/ingestion/<timestamp>.json.
    """
    project_root = Path(__file__).resolve().parents[2]
    log_path_dir = (project_root / log_dir).resolve()
    log_path_dir.mkdir(parents=True, exist_ok=True)

    timestamp    = datetime.now(timezone.utc)
    log_filename = timestamp.strftime("%Y%m%d_%H%M%S") + "_ingestion.json"

    log_data = {
        "timestamp":       timestamp.isoformat(),
        "embedding_model": model_name,
        "collection":      COLLECTION_NAME,
        "source_files":    source_files,
        "total_chunks":    total_chunks,
        "new_chunks":      new_chunks,
        "skipped_chunks":  skipped_chunks,
        "duration_seconds": round(duration_s, 2),
        "chunks_per_second": round(new_chunks / duration_s, 1) if duration_s > 0 else 0,
    }

    with open(log_path_dir / log_filename, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

    logger.info(f"Ingestion log saved → {log_path_dir / log_filename}")


# ── Public API ─────────────────────────────────────────────────────────────────

def embed_and_store(
    chunks: list[dict] | None = None,
    embedding_model: str | None = None,
    vectorstore_path: str | None = None,
) -> dict:
    """
    Embed chunks and upsert them into ChromaDB.

    Args:
        chunks:           List of chunk dicts (from chunker). If None, loads
                          all JSONs from config processed_data_path.
        embedding_model:  Model name. Defaults to config value.
        vectorstore_path: Path to ChromaDB store. Defaults to config value.

    Returns:
        Summary dict: {total, new, skipped, duration_s}
    """
    config = _load_config()
    embedding_model  = embedding_model  or config["embedding_model"]
    vectorstore_path = vectorstore_path or config["vectorstore_path"]
    batch_size       = int(config.get("embedding_batch_size", 64))

    # Load chunks from disk if not passed directly
    if chunks is None:
        logger.info("Loading chunks from data/processed/...")
        chunks = load_processed_chunks(config["processed_data_path"])

    if not chunks:
        raise ValueError("No chunks to embed.")

    # ── Load embedding model ──────────────────────────────────────────────────
    logger.info(f"Loading embedding model: {embedding_model}")
    model = SentenceTransformer(embedding_model)

    # ── Open ChromaDB collection ──────────────────────────────────────────────
    collection = _get_chroma_collection(vectorstore_path)

    # ── Deduplicate — skip chunk_ids already in ChromaDB ─────────────────────
    existing_ids: set[str] = set()
    if collection.count() > 0:
        existing_ids = set(collection.get(include=[])["ids"])

    new_chunks     = [c for c in chunks if c["chunk_id"] not in existing_ids]
    skipped_chunks = len(chunks) - len(new_chunks)

    if skipped_chunks:
        logger.info(f"Skipping {skipped_chunks} already-indexed chunks.")

    if not new_chunks:
        logger.info("All chunks already indexed. Nothing to embed.")
        return {"total": len(chunks), "new": 0, "skipped": skipped_chunks, "duration_s": 0}

    logger.info(f"Embedding {len(new_chunks)} new chunks in batches of {batch_size}...")

    start_time = time.time()

    # ── Batch embed + upsert ──────────────────────────────────────────────────
    for batch_start in tqdm(range(0, len(new_chunks), batch_size), desc="Embedding"):
        batch = new_chunks[batch_start : batch_start + batch_size]

        texts = [c["text"] for c in batch]

        # Embed — returns numpy array of shape (batch_size, embedding_dim)
        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,   # cosine similarity requires unit vectors
        ).tolist()

        # ChromaDB metadata values must be str | int | float | bool
        metadatas = [
            {
                "source_file":         c["source_file"],
                "page_number":         c["page_number"],
                "section_header":      c["section_header"],
                "has_table":           c.get("has_table", False),
                "chunk_size_actual":   c["chunk_size_actual"],
                "ingestion_timestamp": c["ingestion_timestamp"],
            }
            for c in batch
        ]

        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    duration_s = time.time() - start_time

    logger.info(
        f"Ingestion complete — {len(new_chunks)} chunks embedded in "
        f"{duration_s:.1f}s ({len(new_chunks)/duration_s:.0f} chunks/s)"
    )
    logger.info(f"ChromaDB total documents: {collection.count()}")

    # ── Write ingestion log ───────────────────────────────────────────────────
    source_files = sorted({c["source_file"] for c in new_chunks})
    _write_ingestion_log(
        log_dir        = config["ingestion_log_path"],
        total_chunks   = len(chunks),
        new_chunks     = len(new_chunks),
        skipped_chunks = skipped_chunks,
        source_files   = source_files,
        duration_s     = duration_s,
        model_name     = embedding_model,
    )

    return {
        "total":      len(chunks),
        "new":        len(new_chunks),
        "skipped":    skipped_chunks,
        "duration_s": round(duration_s, 2),
    }


# ── Smoke-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = embed_and_store()
    print(f"\nEmbedding summary: {result}")
