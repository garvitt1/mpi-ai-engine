from pathlib import Path

import chromadb

from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SERVICE_DIR = Path(
    "products/04_compliance_and_legal_support"
)

CHROMA_DIR = "data/chroma_db"

COLLECTION_NAME = "mpi_compliance_test"

OLLAMA_MODEL = "llama3.2:3b"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# Configure local models
# --------------------------------------------------

Settings.llm = Ollama(
    model=OLLAMA_MODEL,
    request_timeout=120.0,
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL
)

# Controlled chunking
Settings.node_parser = SentenceSplitter(
    chunk_size=500,
    chunk_overlap=50,
)


# --------------------------------------------------
# Load documents
# --------------------------------------------------

print("\n[1/5] Loading MPI documents...")

documents = SimpleDirectoryReader(
    input_dir=str(SERVICE_DIR),
    recursive=True,
).load_data()

if not documents:
    raise RuntimeError(
        "No MPI documents were found."
    )

print(
    f"Loaded {len(documents)} document(s)."
)


# --------------------------------------------------
# Connect to Chroma
# --------------------------------------------------

print("\n[2/5] Connecting to Chroma...")

client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

# Delete previous test collection if it exists
try:
    client.delete_collection(COLLECTION_NAME)
    print("Removed previous test collection.")
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
# Build index
# --------------------------------------------------

print("\n[3/5] Creating vector index...")

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

print("Vector index created.")


# --------------------------------------------------
# Create query engine
# --------------------------------------------------

print("\n[4/5] Creating RAG query engine...")

query_engine = index.as_query_engine(
    similarity_top_k=3
)

print("RAG query engine ready.")


# --------------------------------------------------
# Ask one question
# --------------------------------------------------

question = (
    "What does the MPI knowledge base say "
    "about GST Filing? Only use information "
    "actually present in the retrieved MPI "
    "documents. If detailed information is "
    "not available, say so clearly."
)

print("\n[5/5] Sending question to local LLM...")
print("\nQUESTION:")
print(question)

response = query_engine.query(question)


# --------------------------------------------------
# Print answer
# --------------------------------------------------

print("\n==============================")
print("MPI RESPONSE")
print("==============================")

print(response)


# --------------------------------------------------
# Print sources
# --------------------------------------------------

print("\n==============================")
print("SOURCES")
print("==============================")

if response.source_nodes:

    seen_sources = set()

    for node in response.source_nodes:

        source = node.node.metadata.get(
            "file_name",
            "Unknown source",
        )

        if source in seen_sources:
            continue

        seen_sources.add(source)

        print(
            f"- {source}"
        )

else:
    print("No sources returned.")