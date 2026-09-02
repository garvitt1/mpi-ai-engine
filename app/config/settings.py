"""
MPI AI Engine configuration.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRODUCTS_DIR = PROJECT_ROOT / "products"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"

COLLECTION_NAME = "mpi_products"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2:3b"

TOP_K_SERVICES = 5
TOP_K_RAG = 10

SEMANTIC_WEIGHT = 0.45
KEYWORD_WEIGHT = 0.55

MIN_RECOMMENDATION_SCORE = 0.30


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