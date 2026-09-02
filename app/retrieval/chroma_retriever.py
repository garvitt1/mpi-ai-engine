"""
MPI Chroma retrieval layer.

Responsibilities:
- Open the existing MPI Chroma database.
- Retrieve knowledge relevant to the customer requirement.
- Accept evidence only from services already selected by
  the deterministic matcher.
- Reject placeholder/template content.
"""

from __future__ import annotations

from typing import Dict, List

import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.config.settings import (
    CHROMA_DIR,
    COLLECTION_NAME,
    TOP_K_RAG,
)


def has_verified_content(text: str) -> bool:
    """
    Determine whether a document contains usable MPI knowledge.

    Placeholder-only service files must not be treated as
    verified factual evidence.
    """

    if not text.strip():
        return False

    placeholder_patterns = [
        "[ADD VERIFIED MPI INFORMATION]",
        "[ADD VERIFIED INFORMATION]",
        "[ADD VERIFIED DETAILS]",
        "[ADD INFORMATION]",
        "TODO",
        "TBD",
    ]

    meaningful_lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        # Ignore markdown headings.
        if line.startswith("#"):
            continue

        lower = line.lower()

        if any(
            placeholder.lower() in lower
            for placeholder in placeholder_patterns
        ):
            continue

        meaningful_lines.append(line)

    return len(meaningful_lines) >= 1


def load_chroma_collection():
    """
    Load the existing MPI Chroma collection.
    """

    if not CHROMA_DIR.exists():
        raise RuntimeError(
            "Chroma database does not exist. "
            "Run: python scripts/ingest.py"
        )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    try:

        return client.get_collection(
            COLLECTION_NAME
        )

    except Exception as exc:

        raise RuntimeError(
            f"Chroma collection '{COLLECTION_NAME}' "
            f"does not exist. "
            f"Run: python scripts/ingest.py"
        ) from exc


def retrieve_evidence(
    requirement: str,
    matched_services: List[Dict],
    embed_model: HuggingFaceEmbedding,
) -> List[Dict]:
    """
    Retrieve evidence for the already-selected MPI services.

    Important:
    The retriever cannot introduce a new service.
    It only searches for evidence belonging to the
    deterministic matcher output.
    """

    collection = load_chroma_collection()

    collection_count = collection.count()

    if collection_count == 0:
        return []

    query_embedding = embed_model.get_text_embedding(
        requirement
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(
            TOP_K_RAG,
            collection_count,
        ),
    )

    documents = results.get(
        "documents",
        [[]],
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]],
    )[0]

    matched_filenames = {
        service["filename"]
        for service in matched_services
    }

    evidence = []

    for document, metadata in zip(
        documents,
        metadatas,
    ):

        metadata = metadata or {}

        filename = metadata.get(
            "filename",
            "",
        )

        # ----------------------------------------------------
        # Critical safety boundary:
        #
        # Evidence must belong to a service already selected
        # by the matcher.
        # ----------------------------------------------------

        if filename not in matched_filenames:
            continue

        if not has_verified_content(document):
            continue

        evidence.append(
            {
                "filename": filename,
                "content": document,
            }
        )

    return evidence