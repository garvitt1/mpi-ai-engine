from pathlib import Path

import chromadb
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PRODUCTS_DIR = Path("products")
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "mpi_products"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# --------------------------------------------------
# Validation
# --------------------------------------------------

if not PRODUCTS_DIR.exists():
    raise FileNotFoundError(
        f"Products directory not found: {PRODUCTS_DIR}"
    )

service_files = sorted(PRODUCTS_DIR.rglob("*.md"))

if not service_files:
    raise RuntimeError(
        "No Markdown service files were found in products/."
    )

print(f"Found {len(service_files)} MPI service files.")


# --------------------------------------------------
# Configure local embedding model
# --------------------------------------------------

print("\nLoading local embedding model...")

Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL
)

Settings.node_parser = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

print("Embedding model ready.")


# --------------------------------------------------
# Load MPI documents
# --------------------------------------------------

print("\nLoading MPI knowledge files...")

documents = SimpleDirectoryReader(
    input_dir=str(PRODUCTS_DIR),
    recursive=True,
    required_exts=[".md"],
    filename_as_id=True,
).load_data()

if not documents:
    raise RuntimeError(
        "Documents were found, but none could be loaded."
    )

print(f"Loaded {len(documents)} documents.")


# --------------------------------------------------
# Add useful metadata
# --------------------------------------------------

for document in documents:
    metadata = document.metadata

    metadata["knowledge_source"] = "MPI Product Knowledge"

    if "file_path" in metadata:
        relative_path = Path(
            metadata["file_path"]
        ).as_posix()

        metadata["source_file"] = relative_path

        parts = Path(
            relative_path
        ).parts

        if len(parts) >= 2:
            metadata["category"] = parts[-2]


# --------------------------------------------------
# Create Chroma database
# --------------------------------------------------

print("\nPreparing Chroma database...")

client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

# Make ingestion reproducible.
# Delete the old MPI collection if it exists,
# then rebuild it from the source files.

try:
    client.delete_collection(COLLECTION_NAME)
    print("Removed previous MPI collection.")
except Exception:
    pass

collection = client.create_collection(
    COLLECTION_NAME
)

vector_store = ChromaVectorStore(
    chroma_collection=collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)


# --------------------------------------------------
# Build vector index
# --------------------------------------------------

print("\nBuilding MPI vector index...")
print("This can take some time on the first run.")

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

print("\n======================================")
print("MPI KNOWLEDGE INDEX CREATED")
print("======================================")
print(f"Documents: {len(documents)}")
print(f"Collection: {COLLECTION_NAME}")
print(f"Database: {CHROMA_DIR}")
print("======================================")