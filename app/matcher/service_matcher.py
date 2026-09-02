"""
MPI service matching.

The matcher is deterministic:
- explicit customer wording is prioritized
- semantic similarity provides additional signal
- the matcher, not the LLM, decides which services are selected
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.config.settings import (
    CATEGORY_CODES,
    KEYWORD_WEIGHT,
    MIN_RECOMMENDATION_SCORE,
    SEMANTIC_WEIGHT,
    TOP_K_SERVICES,
)


SERVICE_KEYWORDS: Dict[str, List[str]] = {

    # IT
    "01_website_development.md": [
        "website",
        "web site",
        "web development",
        "web design",
        "business website",
        "company website",
        "startup website",
        "website development",
        "website design",
        "web application",
    ],

    "02_mobile_app_development.md": [
        "mobile app",
        "android app",
        "ios app",
        "iphone app",
        "app development",
        "mobile application",
    ],

    # MARKETING
    "01_brand_identity_design.md": [
        "branding",
        "brand",
        "brand identity",
        "logo",
        "visual identity",
        "brand design",
        "corporate identity",
    ],

    "02_digital_marketing.md": [
        "digital marketing",
        "online marketing",
        "social media marketing",
        "seo",
        "search engine optimization",
        "online promotion",
        "digital promotion",
    ],

    # COMPLIANCE
    "03_gst_filing.md": [
        "gst",
        "gst filing",
        "gst compliance",
        "gst return",
        "gst returns",
        "goods and services tax",
        "tax compliance",
    ],

    # BUSINESS
    "01_business_registration.md": [
        "business registration",
        "company registration",
        "firm registration",
        "startup registration",
    ],

    "02_udyam_registration_support.md": [
        "udyam",
        "udyam registration",
        "msme registration",
    ],
}


def normalize_text(text: str) -> str:
    """Normalize text for matching."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9₹]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def filename_to_title(filename: str) -> str:
    """Convert a service filename into a readable title."""
    stem = Path(filename).stem
    stem = re.sub(r"^\d+_", "", stem)

    return " ".join(
        word.capitalize()
        for word in stem.replace("_", " ").split()
    )


def get_category_from_path(path: Path) -> str:
    """Return a readable category name from a service path."""
    return path.parent.name.replace("_", " ").title()


def make_service_id(path: Path) -> str:
    """Generate a deterministic MPI service ID."""
    category_folder = path.parent.name

    category_code = CATEGORY_CODES.get(
        category_folder,
        category_folder.upper()[:5],
    )

    match = re.match(r"^(\d+)_", path.name)

    number = int(match.group(1)) if match else 0

    return f"MPI-{category_code}-{number:03d}"


def load_services(products_dir: Path) -> List[Dict]:
    """Load all MPI Markdown service files."""
    files = sorted(products_dir.rglob("*.md"))

    services = []

    for path in files:

        if path.name.lower() in {
            "readme.md",
            "index.md",
        }:
            continue

        try:
            content = path.read_text(
                encoding="utf-8"
            )
        except Exception:
            continue

        services.append(
            {
                "path": path,
                "filename": path.name,
                "name": filename_to_title(path.name),
                "service_id": make_service_id(path),
                "category": get_category_from_path(path),
                "content": content,
            }
        )

    return services


def keyword_score(
    requirement: str,
    service: Dict,
) -> float:
    """
    Calculate an explicit keyword-match score.

    Explicit customer wording is intentionally given
    strong weight because generic semantic similarity can
    otherwise produce unrelated startup services.
    """

    text = normalize_text(requirement)

    keywords = SERVICE_KEYWORDS.get(
        service["filename"],
        [],
    )

    if not keywords:
        return 0.0

    matched = sum(
        1
        for keyword in keywords
        if normalize_text(keyword) in text
    )

    if matched == 0:
        return 0.0

    return min(
        1.0,
        0.55 + (0.15 * matched),
    )


def cosine_similarity(
    a: List[float],
    b: List[float],
) -> float:
    """Calculate cosine similarity between two vectors."""

    numerator = sum(
        x * y
        for x, y in zip(a, b)
    )

    denominator_a = (
        sum(x * x for x in a) ** 0.5
    )

    denominator_b = (
        sum(y * y for y in b) ** 0.5
    )

    if denominator_a == 0 or denominator_b == 0:
        return 0.0

    return numerator / (
        denominator_a * denominator_b
    )


def semantic_score(
    requirement: str,
    service: Dict,
    embed_model: HuggingFaceEmbedding,
) -> float:
    """Calculate semantic similarity between requirement and service."""

    query_embedding = embed_model.get_text_embedding(
        requirement
    )

    searchable_text = (
        f"{service['name']}\n"
        f"{service['category']}\n"
        f"{service['content'][:2000]}"
    )

    service_embedding = embed_model.get_text_embedding(
        searchable_text
    )

    return cosine_similarity(
        query_embedding,
        service_embedding,
    )


def match_services(
    original_requirement: str,
    analysis: str,
    services: List[Dict],
    embed_model: HuggingFaceEmbedding,
) -> List[Dict]:
    """
    Select the most relevant MPI services.

    Explicit keyword matches are combined with semantic
    similarity. Explicit matches are always considered
    strong candidates.
    """

    matching_text = (
        f"{original_requirement}\n{analysis}"
    )

    scored = []

    for service in services:

        key_score = keyword_score(
            matching_text,
            service,
        )

        sem_score = semantic_score(
            matching_text,
            service,
            embed_model,
        )

        final_score = (
            SEMANTIC_WEIGHT * sem_score
            + KEYWORD_WEIGHT * key_score
        )

        item = dict(service)

        item["keyword_score"] = key_score
        item["semantic_score"] = sem_score
        item["score"] = final_score

        scored.append(item)

    scored.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    explicit_matches = [
        item
        for item in scored
        if item["keyword_score"] > 0
    ]

    strong_matches = [
        item
        for item in scored
        if item["score"] >= MIN_RECOMMENDATION_SCORE
    ]

    combined = []
    seen = set()

    for item in explicit_matches + strong_matches:

        filename = item["filename"]

        if filename in seen:
            continue

        seen.add(filename)
        combined.append(item)

    return combined[:TOP_K_SERVICES]