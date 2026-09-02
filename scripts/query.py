import chromadb

from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


# ============================================================
# Configuration
# ============================================================

CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "mpi_products"

OLLAMA_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5


# ============================================================
# Configure local models
# ============================================================

print("Loading local AI models...")

Settings.llm = Ollama(
    model=OLLAMA_MODEL,
    request_timeout=120.0,
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL
)

print("Models ready.")


# ============================================================
# Connect to existing Chroma database
# ============================================================

print("\nConnecting to MPI knowledge base...")

client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

try:
    collection = client.get_collection(
        COLLECTION_NAME
    )
except Exception as error:
    raise RuntimeError(
        f"Chroma collection '{COLLECTION_NAME}' was not found. "
        "Run 'python scripts/ingest.py' first."
    ) from error


print(
    f"Chroma collection found: {COLLECTION_NAME}"
)


# ============================================================
# Connect LlamaIndex to Chroma
# ============================================================

vector_store = ChromaVectorStore(
    chroma_collection=collection
)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store
)

retriever = index.as_retriever(
    similarity_top_k=TOP_K
)


# ============================================================
# System instructions for MPI
# ============================================================

SYSTEM_INSTRUCTIONS = """
You are MPI, an AI assistant for the MPI platform.

Your job is to answer customer questions using ONLY the
MPI knowledge retrieved from the knowledge base.

Rules:
1. Never invent MPI services, capabilities, prices, eligibility,
   turnaround times, SLAs, deliverables, or policies.
2. Do not treat placeholders such as
   "[ADD VERIFIED MPI INFORMATION]" as factual information.
3. If the retrieved information does not answer the question,
   clearly say that the information is not available in the
   current MPI knowledge base.
4. Do not use general world knowledge to fill missing MPI facts.
5. Keep the answer concise and useful.
6. When recommending a service, explain the recommendation
   using the retrieved MPI evidence.
"""


# ============================================================
# Helper: determine whether useful evidence exists
# ============================================================

def has_real_content(results) -> bool:
    """
    Returns False when retrieved content consists mainly of
    template placeholders.
    """

    if not results:
        return False

    useful_markers = [
        "[ADD VERIFIED MPI INFORMATION]",
        "[ADD VERIFIED INFORMATION]",
        "[ADD VERIFIED CUSTOMER SEGMENTS]",
        "[ADD VERIFIED INFORMATION/DOCUMENTS REQUIRED]",
        "[ADD VERIFIED DELIVERABLES]",
        "[ADD VERIFIED PROCESS]",
        "[ADD VERIFIED PRICING]",
        "[ADD VERIFIED TURNAROUND TIME]",
        "[ADD VERIFIED ELIGIBILITY INFORMATION]",
        "[ADD VERIFIED LIMITATIONS]",
        "[ADD VERIFIED ANSWER]",
        "[ADD VERIFIED RELATED MPI SERVICES]",
        "[ADD SEARCH KEYWORDS]",
    ]

    for result in results:

        content = result.node.get_content().strip()

        cleaned = content

        for marker in useful_markers:
            cleaned = cleaned.replace(marker, "")

        if len(cleaned.strip()) > 50:
            return True

    return False


# ============================================================
# Helper: create context
# ============================================================

def build_context(results) -> str:
    context_parts = []

    for number, result in enumerate(results, start=1):

        source = result.node.metadata.get(
            "source_file",
            result.node.metadata.get(
                "file_name",
                "Unknown source"
            )
        )

        content = result.node.get_content().strip()

        context_parts.append(
            f"SOURCE {number}: {source}\n"
            f"{content}"
        )

    return "\n\n".join(context_parts)


# ============================================================
# Start interactive query engine
# ============================================================

print("\n======================================")
print("MPI QUERY ENGINE READY")
print("======================================")
print("Ask an MPI-related question.")
print("Type 'exit' to stop.\n")


while True:

    try:
        question = input("Customer question: ").strip()

    except (KeyboardInterrupt, EOFError):
        print("\nExiting MPI query engine.")
        break

    if question.lower() == "exit":
        print("Exiting MPI query engine.")
        break

    if not question:
        print("Please enter a question.\n")
        continue

    # --------------------------------------------------------
    # Retrieve MPI knowledge
    # --------------------------------------------------------

    print("\nSearching MPI knowledge...")

    try:
        results = retriever.retrieve(question)

    except Exception as error:
        print("\nMPI ERROR")
        print(f"Retrieval failed: {error}\n")
        continue

    # --------------------------------------------------------
    # Check whether useful knowledge exists
    # --------------------------------------------------------

    if not has_real_content(results):

        print("\n======================================")
        print("MPI RESPONSE")
        print("======================================")

        print(
            "The requested information is not available "
            "in the current MPI knowledge base."
        )

        print("\n======================================")
        print("SOURCES")
        print("======================================")

        seen_sources = set()

        for result in results:

            source = result.node.metadata.get(
                "source_file",
                result.node.metadata.get(
                    "file_name",
                    "Unknown source"
                )
            )

            if source not in seen_sources:
                print(f"- {source}")
                seen_sources.add(source)

        print()
        continue

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = build_context(results)

    prompt = f"""
{SYSTEM_INSTRUCTIONS}

RETRIEVED MPI KNOWLEDGE
-----------------------

{context}

CUSTOMER QUESTION
-----------------

{question}

Answer the customer using only the retrieved MPI knowledge.
If the retrieved knowledge is insufficient, say so clearly.
"""

    # --------------------------------------------------------
    # Generate answer with Ollama
    # --------------------------------------------------------

    print("Generating MPI response...")

    try:

        response = Settings.llm.complete(prompt)

        answer = response.text.strip()

    except Exception as error:

        print("\n======================================")
        print("MPI ERROR")
        print("======================================")

        print(
            "The local language model could not generate "
            f"a response: {error}"
        )

        print()

        continue

    # --------------------------------------------------------
    # Print response
    # --------------------------------------------------------

    print("\n======================================")
    print("MPI RESPONSE")
    print("======================================")

    print(answer)

    # --------------------------------------------------------
    # Print sources
    # --------------------------------------------------------

    print("\n======================================")
    print("SOURCES")
    print("======================================")

    seen_sources = set()

    for result in results:

        source = result.node.metadata.get(
            "source_file",
            result.node.metadata.get(
                "file_name",
                "Unknown source"
            )
        )

        if source not in seen_sources:

            print(f"- {source}")

            seen_sources.add(source)

    print()