from pathlib import Path
import re
from typing import Dict, List, Set

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


# ============================================================
# MPI AI ENGINE
# Service Recommendation Engine
# ============================================================
#
# Flow:
#
# Customer Requirement
#        ↓
# Hybrid Service Matching
#        ↓
# Strong Candidate Selection
#        ↓
# Relevant MPI Knowledge Retrieval
#        ↓
# Evidence Filtering
#        ↓
# Local Ollama LLM
#        ↓
# MPI RESPONSE
#
# This version is designed for the local/free MVP.
# ============================================================


# ============================================================
# 1. CONFIGURATION
# ============================================================

PRODUCTS_DIR = Path("products")

CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "mpi_products"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2:3b"

TOP_SERVICES = 5
TOP_K_RAG = 5

SEMANTIC_WEIGHT = 0.65
KEYWORD_WEIGHT = 0.35

# A service must reach this score to be considered
# a meaningful candidate.
MIN_RECOMMENDATION_SCORE = 0.42

# If a direct keyword is found, the service receives
# a strong keyword score.
EXACT_KEYWORD_SCORE = 1.0


# ============================================================
# 2. CATEGORY CODES
# ============================================================

CATEGORY_CODES = {
    "01_packaging_and_printing": "PACK",
    "02_prototyping_and_product_development": "PROTO",
    "03_it_and_digital_services": "IT",
    "04_compliance_and_legal_support": "COMP",
    "05_logistics_and_operations": "LOG",
    "06_marketing_and_sales_support": "MKT",
    "07_business_and_finance_services": "FIN",
    "08_specialized_startup_support": "STARTUP",
}


# ============================================================
# 3. MPI SERVICE KEYWORDS
# ============================================================

SERVICE_KEYWORDS: Dict[str, List[str]] = {

    # --------------------------------------------------------
    # Packaging & Printing
    # --------------------------------------------------------

    "custom_cartons": [
        "custom carton",
        "custom cartons",
        "carton packaging",
    ],

    "corrugated_boxes": [
        "corrugated box",
        "corrugated boxes",
        "shipping box",
        "shipping boxes",
    ],

    "folding_cartons": [
        "folding carton",
        "folding cartons",
    ],

    "rigid_boxes": [
        "rigid box",
        "rigid boxes",
        "premium box",
        "premium boxes",
    ],

    "product_labels": [
        "product label",
        "product labels",
        "label printing",
        "labels",
    ],

    "barcode_stickers": [
        "barcode",
        "barcode sticker",
        "barcode stickers",
    ],

    "tamper_evident_seals": [
        "tamper evident",
        "tamper proof seal",
        "tamper proof seals",
        "tamper seal",
        "tamper seals",
    ],

    "pouches_and_sachets": [
        "pouch",
        "pouches",
        "sachet",
        "sachets",
    ],

    "blister_packaging": [
        "blister packaging",
        "blister pack",
        "blister packs",
    ],

    "shrink_wrapping": [
        "shrink wrapping",
        "shrink wrap",
    ],

    "stretch_film": [
        "stretch film",
        "stretch wrap",
    ],

    "bubble_wrap": [
        "bubble wrap",
        "bubble packaging",
    ],

    "thermal_tags": [
        "thermal tag",
        "thermal tags",
        "thermal label",
        "thermal labels",
    ],

    "instruction_leaflets": [
        "instruction leaflet",
        "instruction leaflets",
        "instruction sheet",
        "product instructions",
    ],

    "brochures_and_catalogs": [
        "brochure",
        "brochures",
        "catalog",
        "catalogue",
        "catalogs",
        "catalogues",
    ],


    # --------------------------------------------------------
    # Prototyping & Product Development
    # --------------------------------------------------------

    "3d_printing": [
        "3d printing",
        "3d print",
        "3d printed",
    ],

    "rapid_prototyping": [
        "rapid prototyping",
        "prototype",
        "prototyping",
    ],

    "cad_design": [
        "cad",
        "cad design",
        "computer aided design",
    ],

    "industrial_design": [
        "industrial design",
        "product design",
    ],

    "model_making": [
        "model making",
        "physical model",
        "prototype model",
    ],

    "electronics_prototyping": [
        "electronics prototype",
        "electronic prototype",
        "electronics prototyping",
    ],

    "mechanical_design": [
        "mechanical design",
        "mechanical engineering",
    ],

    "testing_samples": [
        "testing samples",
        "test samples",
        "sample testing",
    ],


    # --------------------------------------------------------
    # IT & Digital Services
    # --------------------------------------------------------

    "website_development": [
        "website",
        "web development",
        "website development",
        "business website",
        "company website",
        "online presence",
        "web portal",
        "web application",
    ],

    "mobile_app_development": [
        "mobile app",
        "mobile application",
        "android app",
        "ios app",
        "app development",
    ],

    "ui_ux_design": [
        "ui",
        "ux",
        "ui ux",
        "user interface",
        "user experience",
        "interface design",
        "app design",
        "website design",
    ],

    "erp_setup": [
        "erp",
        "erp setup",
        "enterprise resource planning",
    ],

    "crm_setup": [
        "crm",
        "crm setup",
        "customer relationship management",
    ],

    "software_qa_testing": [
        "software testing",
        "software qa",
        "qa testing",
        "software quality testing",
    ],

    "cloud_hosting_setup": [
        "cloud hosting",
        "cloud setup",
        "hosting",
        "server hosting",
    ],

    "cybersecurity_audit": [
        "cybersecurity",
        "cyber security",
        "security audit",
        "cybersecurity audit",
    ],

    "api_integration": [
        "api integration",
        "api",
        "system integration",
        "software integration",
    ],

    "data_entry_and_digitization": [
        "data entry",
        "digitization",
        "data digitization",
        "document digitization",
    ],


    # --------------------------------------------------------
    # Compliance & Legal Support
    # --------------------------------------------------------

    "company_incorporation_support": [
        "company registration",
        "company incorporation",
        "incorporation",
        "register a company",
    ],

    "udyam_registration_support": [
        "udyam",
        "udyam registration",
        "msme registration",
    ],

    "gst_filing": [
        "gst",
        "gst filing",
        "gst return",
        "gst returns",
        "gst compliance",
    ],

    "tds_filing": [
        "tds",
        "tds filing",
        "tds return",
        "tds returns",
    ],

    "roc_compliance": [
        "roc",
        "roc compliance",
        "mca compliance",
        "company compliance",
    ],

    "trademark_filing": [
        "trademark",
        "trademark registration",
        "brand registration",
    ],

    "copyright_filing": [
        "copyright",
        "copyright registration",
    ],

    "patent_drafting_support": [
        "patent",
        "patent drafting",
        "patent application",
    ],

    "iso_certification_support": [
        "iso",
        "iso certification",
        "iso certification support",
    ],

    "zed_certification_support": [
        "zed",
        "zed certification",
    ],


    # --------------------------------------------------------
    # Logistics & Operations
    # --------------------------------------------------------

    "local_transportation": [
        "local transportation",
        "local transport",
    ],

    "courier_services": [
        "courier",
        "courier service",
        "courier services",
        "parcel delivery",
    ],

    "warehouse_rental": [
        "warehouse",
        "warehouse rental",
        "storage space",
    ],

    "packaging_fulfillment": [
        "fulfillment",
        "packaging fulfillment",
        "order fulfillment",
    ],

    "last_mile_delivery": [
        "last mile",
        "last-mile delivery",
        "last mile delivery",
    ],

    "inventory_management": [
        "inventory",
        "inventory management",
        "stock management",
    ],

    "order_tracking_setup": [
        "order tracking",
        "shipment tracking",
        "tracking system",
    ],

    "reverse_logistics": [
        "reverse logistics",
        "returns logistics",
        "product returns",
    ],

    "loading_and_unloading": [
        "loading",
        "unloading",
        "loading and unloading",
    ],

    "cold_chain_handling": [
        "cold chain",
        "temperature controlled logistics",
        "cold storage transport",
    ],


    # --------------------------------------------------------
    # Marketing & Sales Support
    # --------------------------------------------------------

    "brand_identity_design": [
        "branding",
        "brand identity",
        "brand design",
        "brand guidelines",
    ],

    "logo_design": [
        "logo",
        "brand logo",
        "logo design",
    ],

    "social_media_management": [
        "social media",
        "instagram",
        "facebook",
        "linkedin",
        "social media management",
    ],

    "performance_marketing": [
        "performance marketing",
        "digital advertising",
        "online advertising",
        "paid advertising",
    ],

    "product_photography": [
        "product photography",
        "product photos",
        "product photography service",
    ],

    "video_editing": [
        "video editing",
        "video editor",
        "edit videos",
    ],

    "catalog_design": [
        "catalog design",
        "catalogue design",
    ],

    "sales_deck_design": [
        "sales deck",
        "sales presentation",
        "pitch deck",
        "presentation design",
    ],

    "market_research": [
        "market research",
        "market analysis",
        "customer research",
        "market study",
    ],

    "lead_generation": [
        "lead generation",
        "generate leads",
        "sales leads",
        "customer leads",
    ],


    # --------------------------------------------------------
    # Business & Finance
    # --------------------------------------------------------

    "accounting": [
        "accounting",
        "financial accounting",
        "accounts",
    ],

    "bookkeeping": [
        "bookkeeping",
        "book keeping",
        "accounts maintenance",
    ],

    "mis_reporting": [
        "mis reporting",
        "management reporting",
        "mis report",
    ],

    "cma_preparation": [
        "cma",
        "cma preparation",
    ],

    "project_report_preparation": [
        "project report",
        "project report preparation",
        "business project report",
    ],

    "valuation_support": [
        "valuation",
        "business valuation",
        "company valuation",
    ],

    "due_diligence_support": [
        "due diligence",
        "business due diligence",
        "financial due diligence",
    ],

    "business_analysis": [
        "business analysis",
        "business analytics",
        "business analysis support",
    ],

    "payroll_processing": [
        "payroll",
        "payroll processing",
        "salary processing",
    ],

    "virtual_cfo_services": [
        "virtual cfo",
        "fractional cfo",
        "cfo services",
        "financial strategy",
    ],


    # --------------------------------------------------------
    # Specialized Startup Support
    # --------------------------------------------------------

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
        "product samples",
        "source samples",
    ],

    "toolroom_support": [
        "toolroom",
        "tool room",
        "toolroom support",
    ],

    "packaging_design_consultation": [
        "packaging design",
        "packaging consultation",
        "package design",
    ],

    "procurement_advisory": [
        "procurement",
        "procurement advisory",
        "purchasing",
        "procurement support",
    ],

    "vendor_onboarding": [
        "vendor onboarding",
        "supplier onboarding",
        "vendor registration",
    ],

    "b2b_sourcing_coordination": [
        "b2b sourcing",
        "supplier sourcing",
        "business suppliers",
        "find suppliers",
        "supplier search",
        "vendor sourcing",
    ],

    "custom_small_batch_manufacturing": [
        "small batch",
        "small batch manufacturing",
        "small quantity manufacturing",
    ],
}


# ============================================================
# 4. TEXT UTILITIES
# ============================================================

def normalize(text: str) -> str:
    """Normalize text for matching."""

    return re.sub(
        r"\s+",
        " ",
        text.lower().strip(),
    )


def tokenize(text: str) -> Set[str]:
    """Convert text into normalized tokens."""

    return set(
        re.findall(
            r"[a-z0-9]+",
            normalize(text),
        )
    )


# ============================================================
# 5. SERVICE METADATA
# ============================================================

def get_service_key(filename: str) -> str:
    """01_website_development.md -> website_development"""

    name = Path(filename).stem

    return re.sub(
        r"^\d+_",
        "",
        name,
    )


def get_service_name(filename: str) -> str:
    """01_website_development.md -> Website Development"""

    key = get_service_key(filename)

    return key.replace(
        "_",
        " ",
    ).title()


def get_category_name(directory_name: str) -> str:
    """
    03_it_and_digital_services
    -> It And Digital Services
    """

    name = re.sub(
        r"^\d+_",
        "",
        directory_name,
    )

    return name.replace(
        "_",
        " ",
    ).title()


def get_service_id(
    category_dir: Path,
    filename: str,
) -> str:
    """Generate a deterministic MPI service ID."""

    code = CATEGORY_CODES.get(
        category_dir.name
    )

    if code is None:
        raise ValueError(
            f"Unknown MPI category: {category_dir.name}"
        )

    match = re.match(
        r"^(\d+)_",
        filename,
    )

    if not match:
        raise ValueError(
            f"Invalid service filename: {filename}"
        )

    number = match.group(1).zfill(3)

    return f"MPI-{code}-{number}"


# ============================================================
# 6. KEYWORD MATCHING
# ============================================================

def calculate_keyword_score(
    query: str,
    service_key: str,
    service_name: str,
) -> float:

    query_normalized = normalize(query)

    keywords = SERVICE_KEYWORDS.get(
        service_key,
        [normalize(service_name)],
    )

    # Exact phrase match gets maximum keyword score.
    for keyword in keywords:

        if normalize(keyword) in query_normalized:
            return EXACT_KEYWORD_SCORE

    query_tokens = tokenize(query)

    keyword_tokens: Set[str] = set()

    for keyword in keywords:
        keyword_tokens.update(
            tokenize(keyword)
        )

    if not query_tokens or not keyword_tokens:
        return 0.0

    overlap = (
        query_tokens.intersection(
            keyword_tokens
        )
    )

    return min(
        len(overlap) / len(keyword_tokens),
        1.0,
    )


# ============================================================
# 7. LOAD ALL MPI SERVICES
# ============================================================

def load_services() -> List[Dict[str, str]]:

    if not PRODUCTS_DIR.exists():

        raise FileNotFoundError(
            f"MPI products directory not found: "
            f"{PRODUCTS_DIR}"
        )

    services = []

    for file_path in sorted(
        PRODUCTS_DIR.rglob("*.md")
    ):

        relative_parts = (
            file_path.relative_to(
                PRODUCTS_DIR
            ).parts
        )

        # A valid service must be:
        # products/category/service.md
        if len(relative_parts) != 2:
            continue

        category_dir = file_path.parent

        service_key = get_service_key(
            file_path.name
        )

        service_name = get_service_name(
            file_path.name
        )

        category = get_category_name(
            category_dir.name
        )

        service_id_value = get_service_id(
            category_dir,
            file_path.name
        )

        services.append(
            {
                "service_key": service_key,
                "service_name": service_name,
                "category": category,
                "service_id": service_id_value,
                "file": file_path.as_posix(),
            }
        )

    return services


print("\nLoading MPI services...")

services = load_services()

if len(services) != 83:

    raise RuntimeError(
        f"Expected exactly 83 MPI services, "
        f"but found {len(services)}. "
        "Check the products/ directory."
    )

print(
    f"Loaded {len(services)} MPI services."
)


# ============================================================
# 8. LOAD LOCAL EMBEDDING MODEL
# ============================================================

print("\nLoading local embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model ready.")


# ============================================================
# 9. BUILD SERVICE EMBEDDINGS
# ============================================================

print("\nCreating service embeddings...")

service_texts = []

for service in services:

    keywords = SERVICE_KEYWORDS.get(
        service["service_key"],
        [],
    )

    keyword_text = " ".join(
        keywords
    )

    service_texts.append(
        (
            f"Service: {service['service_name']}. "
            f"Category: {service['category']}. "
            f"Keywords: {keyword_text}."
        )
    )


service_embeddings = embedding_model.encode(
    service_texts,
    normalize_embeddings=True,
    show_progress_bar=False,
)

print(
    f"Created embeddings for "
    f"{len(service_embeddings)} services."
)


# ============================================================
# 10. CONFIGURE LOCAL OLLAMA
# ============================================================

Settings.llm = Ollama(
    model=OLLAMA_MODEL,
    request_timeout=120.0,
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL,
)


# ============================================================
# 11. CONNECT TO CHROMA
# ============================================================

print("\nConnecting to MPI knowledge base...")

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

try:

    collection = chroma_client.get_collection(
        COLLECTION_NAME
    )

except Exception as error:

    raise RuntimeError(
        f"Chroma collection '{COLLECTION_NAME}' "
        "was not found. "
        "Run 'python scripts/ingest.py' first."
    ) from error


if collection.count() == 0:

    raise RuntimeError(
        "MPI Chroma collection is empty. "
        "Run 'python scripts/ingest.py' again."
    )


vector_store = ChromaVectorStore(
    chroma_collection=collection
)

rag_index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store
)

rag_retriever = rag_index.as_retriever(
    similarity_top_k=TOP_K_RAG
)

print(
    f"MPI knowledge base ready. "
    f"Indexed chunks: {collection.count()}"
)


# ============================================================
# 12. PLACEHOLDER DETECTION
# ============================================================

PLACEHOLDER_PATTERNS = [
    r"\[ADD VERIFIED MPI INFORMATION\]",
    r"\[ADD VERIFIED INFORMATION\]",
    r"\[ADD VERIFIED CUSTOMER SEGMENTS\]",
    r"\[ADD VERIFIED INFORMATION/DOCUMENTS REQUIRED\]",
    r"\[ADD VERIFIED DELIVERABLES\]",
    r"\[ADD VERIFIED PROCESS\]",
    r"\[ADD VERIFIED PRICING\]",
    r"\[ADD VERIFIED TURNAROUND TIME\]",
    r"\[ADD VERIFIED ELIGIBILITY INFORMATION\]",
    r"\[ADD VERIFIED LIMITATIONS\]",
    r"\[ADD VERIFIED ANSWER\]",
    r"\[ADD VERIFIED RELATED MPI SERVICES\]",
    r"\[ADD SEARCH KEYWORDS\]",
]


def clean_document_text(text: str) -> str:
    """Remove placeholder content from a document."""

    cleaned = text

    for pattern in PLACEHOLDER_PATTERNS:

        cleaned = re.sub(
            pattern,
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

    # Remove Markdown headings.
    cleaned = re.sub(
        r"^\s*#+\s*",
        "",
        cleaned,
        flags=re.MULTILINE,
    )

    # Normalize whitespace.
    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned,
    ).strip()

    return cleaned


def has_verified_content(text: str) -> bool:
    """
    Determine whether a document contains meaningful
    non-placeholder content.
    """

    if not text:
        return False

    cleaned = clean_document_text(
        text
    )

    # At the current stage, files that contain only
    # headings/metadata/source information are not treated
    # as substantive knowledge.
    if len(cleaned) < 120:
        return False

    # If practically everything is placeholder material,
    # reject it.
    placeholder_count = 0

    for pattern in PLACEHOLDER_PATTERNS:

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            placeholder_count += 1

    # A heavily templated document is not considered
    # verified knowledge.
    if placeholder_count >= 3 and len(cleaned) < 250:
        return False

    return True


# ============================================================
# 13. SOURCE PATH NORMALIZATION
# ============================================================

def normalize_source_path(
    source: str,
) -> str:

    path = Path(
        str(source)
    ).as_posix()

    if "products/" in path:

        path = (
            "products/"
            + path.split(
                "products/",
                1,
            )[1]
        )

    return path


# ============================================================
# 14. RETRIEVE RELEVANT MPI EVIDENCE
# ============================================================

def retrieve_evidence(
    question: str,
    selected_matches: List[Dict],
) -> str:

    if not selected_matches:
        return ""

    allowed_files = {
        item["service"]["file"]
        for item in selected_matches
    }

    try:

        results = rag_retriever.retrieve(
            question
        )

    except Exception as error:

        print(
            f"Warning: MPI knowledge retrieval failed: "
            f"{error}"
        )

        return ""

    evidence_parts = []

    seen_sources: Set[str] = set()

    for result in results:

        metadata = result.node.metadata

        source = metadata.get(
            "source_file",
            metadata.get(
                "file_path",
                metadata.get(
                    "file_name",
                    "",
                ),
            ),
        )

        if not source:
            continue

        normalized_source = normalize_source_path(
            str(source)
        )

        if normalized_source not in allowed_files:
            continue

        if normalized_source in seen_sources:
            continue

        content = result.node.get_content().strip()

        if not has_verified_content(
            content
        ):
            continue

        seen_sources.add(
            normalized_source
        )

        evidence_parts.append(
            f"SOURCE: {normalized_source}\n"
            f"{clean_document_text(content)}"
        )

    return "\n\n".join(
        evidence_parts
    )


# ============================================================
# 15. GENERATE GROUNDED RESPONSE
# ============================================================

def generate_response(
    question: str,
    selected_matches: List[Dict],
    evidence: str,
) -> str:

    service_context = "\n".join(
        [
            (
                f"{index}. "
                f"{item['service']['service_name']} "
                f"| Service ID: "
                f"{item['service']['service_id']} "
                f"| Category: "
                f"{item['service']['category']}"
            )
            for index, item in enumerate(
                selected_matches,
                start=1,
            )
        ]
    )

    if evidence:

        evidence_context = evidence

    else:

        evidence_context = (
            "NO VERIFIED DETAILED MPI KNOWLEDGE "
            "WAS RETRIEVED."
        )

    prompt = f"""
You are MPI, an AI assistant for the MPI platform.

Your job is to identify the customer's requirement and
recommend only relevant MPI services.

CUSTOMER REQUIREMENT
====================
{question}

SERVICES SELECTED BY THE MPI MATCHING SYSTEM
=============================================
{service_context}

VERIFIED MPI EVIDENCE
=====================
{evidence_context}

STRICT RULES
============

1. Treat the MPI service list as authoritative for which
   services exist.

2. Treat verified retrieved MPI evidence as authoritative
   for detailed claims about those services.

3. Never invent MPI capabilities.

4. Never invent or assume:
   - pricing
   - loans
   - funding
   - credit facilities
   - eligibility
   - turnaround times
   - SLAs
   - deliverables
   - certifications
   - guarantees
   - government approvals
   - processes

5. General knowledge about an industry service is NOT evidence
   that MPI provides that capability.

6. Placeholder text is NOT information.

7. Do not convert missing information into a factual statement.

8. Do not introduce a service that is not in the selected
   MPI services.

9. Do not reorder the services based on your own reasoning.
   Preserve their supplied order.

10. If no detailed MPI evidence is available, explicitly say
    that detailed information is not available in the current
    MPI knowledge base.

11. Do not create a recommendation merely to fill the response.

12. For unsupported requests such as loans or funding, do not
    connect unrelated MPI services merely because they might
    be useful in the general world.

13. Never invent a service ID. Use only the IDs supplied above.

14. Keep the answer concise and factual.

Return exactly this format:

Customer Requirement
[brief summary]

Recommended MPI Services
[list only the clearly relevant selected services]

Why These Services
[brief explanation based on the customer's requirement]

Available MPI Information
[only information explicitly supported by verified MPI evidence]

Information Not Available
[important information that is missing]

Sources
[list only the source files actually used]
"""

    print("\nGenerating MPI response...")

    try:

        response = Settings.llm.complete(
            prompt
        )

    except Exception as error:

        raise RuntimeError(
            f"Local Ollama generation failed: {error}"
        ) from error

    answer = response.text.strip()

    if not answer:

        raise RuntimeError(
            "Ollama returned an empty response."
        )

    return answer


# ============================================================
# 16. INTERACTIVE ENGINE
# ============================================================

print("\n======================================")
print("MPI SERVICE RECOMMENDATION ENGINE")
print("======================================")
print("Describe the customer's requirement.")
print("Type 'exit' to stop.\n")


while True:

    try:

        question = input(
            "Customer requirement: "
        ).strip()

    except (
        KeyboardInterrupt,
        EOFError,
    ):

        print(
            "\nExiting MPI recommendation engine."
        )

        break


    if question.lower() == "exit":

        print(
            "Exiting MPI recommendation engine."
        )

        break


    if not question:

        print(
            "Please enter a customer requirement.\n"
        )

        continue


    # ========================================================
    # STEP A — CREATE CUSTOMER EMBEDDING
    # ========================================================

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )


    # ========================================================
    # STEP B — SEMANTIC SIMILARITY
    # ========================================================

    semantic_scores = cosine_similarity(
        query_embedding,
        service_embeddings,
    )[0]


    # ========================================================
    # STEP C — HYBRID MATCHING
    # ========================================================

    candidates = []

    for index, service in enumerate(
        services
    ):

        semantic_score = float(
            semantic_scores[index]
        )

        keyword_score = calculate_keyword_score(
            question,
            service["service_key"],
            service["service_name"],
        )

        combined_score = (
            semantic_score
            * SEMANTIC_WEIGHT
            +
            keyword_score
            * KEYWORD_WEIGHT
        )

        candidates.append(
            {
                "service": service,
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "score": combined_score,
            }
        )


    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )


    # ========================================================
    # STEP D — STRONG MATCH FILTER
    # ========================================================

    strong_matches = [
        candidate
        for candidate in candidates
        if candidate["score"]
        >= MIN_RECOMMENDATION_SCORE
    ]


    # ========================================================
    # STEP E — NO-MATCH SAFETY
    # ========================================================

    if not strong_matches:

        print(
            "\n======================================"
        )

        print(
            "MPI RESPONSE"
        )

        print(
            "======================================"
        )

        print(
            "No suitable MPI service was identified "
            "for this requirement from the current "
            "MPI service catalog."
        )

        print()

        continue


    # Keep only the strongest candidates.
    selected_matches = strong_matches[
        :TOP_SERVICES
    ]


    # ========================================================
    # STEP F — RETRIEVE SUPPORTING EVIDENCE
    # ========================================================

    print(
        "\nSearching MPI knowledge..."
    )

    evidence = retrieve_evidence(
        question,
        selected_matches,
    )


    # ========================================================
    # STEP G — GENERATE RESPONSE
    # ========================================================

    try:

        answer = generate_response(
            question,
            selected_matches,
            evidence,
        )

    except RuntimeError as error:

        print(
            "\n======================================"
        )

        print(
            "MPI ERROR"
        )

        print(
            "======================================"
        )

        print(error)

        print()

        continue


    # ========================================================
    # STEP H — DISPLAY RESPONSE
    # ========================================================

    print(
        "\n======================================"
    )

    print(
        "MPI RESPONSE"
    )

    print(
        "======================================"
    )

    print(answer)

    print()