from pathlib import Path

import chromadb
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SERVICE_FILE = Path(
    "products/04_compliance_and_legal_support/03_gst_filing.md"
)

CHROMA_PATH = "data/chroma_db"

COLLECTION_NAME = "mpi_test_gst_filing"


# --------------------------------------------------
# Local embedding model
# --------------------------------------------------

print("Loading embedding model...")

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# --------------------------------------------------
# Load the MPI service document
# --------------------------------------------------

print("Loading MPI service document...")

documents = SimpleDirectoryReader(
    input_files=[str(SERVICE_FILE)]
).load_data()

print(f"Loaded {len(documents)} document(s).")


# --------------------------------------------------
# Create Chroma collection
# --------------------------------------------------

print("Connecting to Chroma...")

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_or_create_collection(
    COLLECTION_NAME
)

vector_store = ChromaVectorStore(
    chroma_collection=collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)


# --------------------------------------------------
# Build the vector index
# --------------------------------------------------

print("Creating vector index...")

VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

print("Successfully indexed GST Filing.")

print(f"Chroma collection: {COLLECTION_NAME}")
print(f"Stored at: {CHROMA_PATH}")