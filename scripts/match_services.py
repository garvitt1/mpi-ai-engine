from pathlib import Path
import re

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Configuration
# ============================================================

PRODUCTS_DIR = Path("products")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 8

# Weighting:
# Semantic similarity understands meaning.
# Keyword matching protects exact service intent.
SEMANTIC_WEIGHT = 0.65
KEYWORD_WEIGHT = 0.35


# ============================================================
# Service-specific keyword hints
# ============================================================

SERVICE_KEYWORDS = {
    "website_development": [
        "website",
        "web development",
        "web application",
        "business website",
        "company website",
        "online presence",
        "web page",
        "web portal",
    ],

    "mobile_app_development": [
        "mobile app",
        "android app",
        "ios app",
        "application",
        "mobile application",
    ],

    "ui_ux_design": [
        "ui",
        "ux",
        "user interface",
        "user experience",
        "interface design",
        "app design",
        "website design",
    ],

    "logo_design": [
        "logo",
        "brand logo",
        "logo design",
    ],

    "brand_identity_design": [
        "branding",
        "brand identity",
        "brand design",
        "brand guidelines",
    ],

    "gst_filing": [
        "gst",
        "gst filing",
        "gst return",
        "gst compliance",
        "gst returns",
    ],

    "trademark_filing": [
        "trademark",
        "trademark registration",
        "brand registration",
    ],

    "accounting": [
        "accounting",
        "accounts",
        "financial accounting",
    ],

    "bookkeeping": [
        "bookkeeping",
        "book keeping",
        "accounts maintenance",
    ],

    "market_research": [
        "market research",
        "market analysis",
        "customer research",
        "market study",
    ],

    "lead_generation": [
        "lead generation",
        "leads",
        "customer leads",
        "sales leads",
    ],

    "social_media_management": [
        "social media",
        "instagram",
        "facebook",
        "linkedin",
        "social media management",
    ],

    "b2b_sourcing_coordination": [
        "b2b sourcing",
        "supplier sourcing",
        "business suppliers",
        "find suppliers",
        "supplier search",
        "vendor sourcing",
    ],

    "vendor_onboarding": [
        "vendor onboarding",
        "supplier onboarding",
        "vendor registration",
    ],

    "procurement_advisory": [
        "procurement",
        "procurement advisory",
        "purchasing",
        "procurement support",
    ],

    "lab_testing": [
        "lab testing",
        "laboratory testing",
        "testing laboratory",
    ],

    "quality_assurance": [
        "quality assurance",
        "qa",
        "quality check",
        "quality testing",
    ],

    "certification_testing": [
        "certification testing",
        "certification",
        "compliance testing",
    ],

    "sample_sourcing": [
        "sample sourcing",
        "product sample",
        "samples",
        "source samples",
    ],
}


# ============================================================
# Helper functions
# ============================================================

def clean_service_name(filename: str) -> str:
    """
    Convert:
        01_website_development.md

    into:
        Website Development
    """

    name = Path(filename).stem

    name = re.sub(r"^\d+_", "", name)

    return name.replace("_", " ").lower()


def display_name(name: str) -> str:
    return name.replace("_", " ").title()


def clean_category(directory_name: str) -> str:
    """
    Convert:
        03_it_and_digital_services

    into:
        It And Digital Services
    """

    name = re.sub(
        r"^\d+_",
        "",
        directory_name
    )

    return name.replace("_", " ").lower()


def service_id(category_dir: Path, filename: str) -> str:

    category_codes = {
        "01_packaging_and_printing": "PACK",
        "02_prototyping_and_product_development": "PROTO",
        "03_it_and_digital_services": "IT",
        "04_compliance_and_legal_support": "COMP",
        "05_logistics_and_operations": "LOG",
        "06_marketing_and_sales_support": "MKT",
        "07_business_and_finance_services": "FIN",
        "08_specialized_startup_support": "STARTUP",
    }

    code = category_codes.get(
        category_dir.name,
        "MPI"
    )

    match = re.match(
        r"^(\d+)_",
        filename
    )

    number = (
        match.group(1).zfill(3)
        if match
        else "000"
    )

    return f"MPI-{code}-{number}"


def tokenize(text: str) -> set[str]:
    """
    Convert text into normalized tokens.
    """

    return set(
        re.findall(
            r"[a-z0-9]+",
            text.lower()
        )
    )


def keyword_score(
    query: str,
    service_key: str,
    service_name: str
) -> float:

    query_lower = query.lower()

    keywords = SERVICE_KEYWORDS.get(
        service_key,
        []
    )

    if not keywords:
        keywords = [
            service_name.replace("_", " ")
        ]

    # Strong exact phrase matching
    exact_matches = 0

    for keyword in keywords:

        if keyword in query_lower:
            exact_matches += 1

    if exact_matches > 0:
        return 1.0

    # Token overlap
    query_tokens = tokenize(query)

    candidate_tokens = set()

    for keyword in keywords:
        candidate_tokens.update(
            tokenize(keyword)
        )

    if not query_tokens or not candidate_tokens:
        return 0.0

    overlap = (
        query_tokens.intersection(
            candidate_tokens
        )
    )

    return len(overlap) / len(candidate_tokens)


# ============================================================
# Load services
# ============================================================

print("\nLoading MPI services...")

services = []

for file_path in sorted(
    PRODUCTS_DIR.rglob("*.md")
):

    if len(file_path.parts) < 3:
        continue

    category_dir = file_path.parent

    service_key = Path(
        file_path.name
    ).stem

    service_key = re.sub(
        r"^\d+_",
        "",
        service_key
    )

    name = clean_service_name(
        file_path.name
    )

    category = clean_category(
        category_dir.name
    )

    services.append(
        {
            "service_key": service_key,
            "service_name": name,
            "category": category,
            "service_id": service_id(
                category_dir,
                file_path.name
            ),
            "file": str(file_path),
        }
    )


if not services:
    raise RuntimeError(
        "No MPI service files were found."
    )


print(
    f"Loaded {len(services)} MPI services."
)


# ============================================================
# Load embedding model
# ============================================================

print("\nLoading local embedding model...")

model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model ready.")


# ============================================================
# Create service representations
# ============================================================

service_texts = []

for service in services:

    keywords = SERVICE_KEYWORDS.get(
        service["service_key"],
        []
    )

    keyword_text = " ".join(
        keywords
    )

    text = (
        f"Service: "
        f"{service['service_name']}. "
        f"Category: "
        f"{service['category']}. "
        f"Keywords: "
        f"{keyword_text}"
    )

    service_texts.append(text)


print("\nCreating service embeddings...")

service_embeddings = model.encode(
    service_texts,
    normalize_embeddings=True,
    show_progress_bar=False,
)

print(
    f"Created embeddings for "
    f"{len(service_embeddings)} services."
)


# ============================================================
# Interactive matcher
# ============================================================

print("\n======================================")
print("MPI CUSTOMER REQUIREMENT MATCHER")
print("======================================")
print("Describe what the customer needs.")
print("Type 'exit' to stop.\n")


while True:

    question = input(
        "Customer requirement: "
    ).strip()

    if question.lower() == "exit":

        print("Exiting matcher.")

        break

    if not question:

        print(
            "Please enter a customer requirement.\n"
        )

        continue

    # --------------------------------------------------------
    # Semantic similarity
    # --------------------------------------------------------

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
    )

    semantic_scores = cosine_similarity(
        query_embedding,
        service_embeddings,
    )[0]

    # --------------------------------------------------------
    # Hybrid score
    # --------------------------------------------------------

    results = []

    for index, service in enumerate(
        services
    ):

        semantic_score = float(
            semantic_scores[index]
        )

        keyword_score_value = keyword_score(
            question,
            service["service_key"],
            service["service_name"],
        )

        combined_score = (
            semantic_score
            * SEMANTIC_WEIGHT
            +
            keyword_score_value
            * KEYWORD_WEIGHT
        )

        results.append(
            {
                "service": service,
                "semantic_score": semantic_score,
                "keyword_score": keyword_score_value,
                "combined_score": combined_score,
            }
        )

    results.sort(
        key=lambda item: item["combined_score"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print(
        "\n======================================"
    )

    print(
        "RELEVANT MPI SERVICES"
    )

    print(
        "======================================"
    )

    for rank, result in enumerate(
        results[:TOP_K],
        start=1
    ):

        service = result["service"]

        print(
            f"\n{rank}. "
            f"{display_name(service['service_name'])}"
        )

        print(
            f"   Category: "
            f"{display_name(service['category'])}"
        )

        print(
            f"   Service ID: "
            f"{service['service_id']}"
        )

        print(
            f"   Hybrid score: "
            f"{result['combined_score']:.4f}"
        )

        print(
            f"   Semantic: "
            f"{result['semantic_score']:.4f}"
        )

        print(
            f"   Keyword: "
            f"{result['keyword_score']:.4f}"
        )

        print(
            f"   Source: "
            f"{service['file']}"
        )

    print()