from pathlib import Path

import chromadb
from llama_index.core import (
    Settings,
    VectorStoreIndex,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "mpi_test_gst_filing"


# --------------------------------------------------
# Load the same embedding model
# --------------------------------------------------

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# Connect to existing Chroma collection
# --------------------------------------------------

print("Connecting to Chroma...")

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_collection(
    COLLECTION_NAME
)

vector_store = ChromaVectorStore(
    chroma_collection=collection
)


# --------------------------------------------------
# Load index from Chroma
# --------------------------------------------------

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store
)


# --------------------------------------------------
# Create retriever
# --------------------------------------------------

retriever = index.as_retriever(
    similarity_top_k=3
)


# --------------------------------------------------
# Test query
# --------------------------------------------------

question = "What is GST Filing?"

print("\nCustomer Question:")
print(question)

print("\nSearching MPI knowledge...")

results = retriever.retrieve(question)


# --------------------------------------------------
# Display retrieved information
# --------------------------------------------------

print("\n--- RETRIEVED INFORMATION ---")

if not results:
    print("No results found.")
else:
    for number, result in enumerate(results, start=1):

        print(f"\nResult {number}")
        print("-" * 40)

        print("Similarity Score:", result.score)

        print("Source:")

        print(
            result.node.metadata.get(
                "file_name",
                "Unknown"
            )
        )

        print("\nContent:")
        print(result.node.get_content())