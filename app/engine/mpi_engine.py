"""
Main MPI AI Engine.

Coordinates:
1. Requirement analysis
2. Deterministic service matching
3. MPI knowledge retrieval
4. Evidence validation
5. Final response construction
"""

from __future__ import annotations

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from app.analyzer.requirement_analyzer import (
    analyze_requirement,
)
from app.config.settings import (
    EMBEDDING_MODEL,
    OLLAMA_MODEL,
    PRODUCTS_DIR,
)
from app.matcher.service_matcher import (
    load_services,
    match_services,
)
from app.rag.evidence_validator import (
    build_response,
)
from app.retrieval.chroma_retriever import (
    retrieve_evidence,
)


class MPIEngine:
    """Coordinates the complete MPI AI pipeline."""

    def __init__(self) -> None:

        print("Loading local models...")

        self.embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL
        )

        self.llm = Ollama(
            model=OLLAMA_MODEL,
            request_timeout=120.0,
        )

        print("Local models ready.")

        self.services = load_services(
            PRODUCTS_DIR
        )

        if not self.services:
            raise RuntimeError(
                "No MPI service files found in products/."
            )

        print(
            f"Loaded {len(self.services)} MPI service files."
        )

    def run(
        self,
        customer_requirement: str,
    ) -> str:
        """
        Run the complete MPI AI Engine pipeline.
        """

        customer_requirement = (
            customer_requirement.strip()
        )

        if not customer_requirement:
            return (
                "Please provide a customer requirement."
            )

        print()
        print("=" * 60)
        print("MPI AI ENGINE")
        print("=" * 60)

        # ====================================================
        # STEP 1: REQUIREMENT ANALYSIS
        # ====================================================

        print()
        print("STEP 1: REQUIREMENT ANALYSIS")

        analysis = analyze_requirement(
            customer_requirement,
            self.llm,
        )

        print(analysis)

        # ====================================================
        # STEP 2: SERVICE MATCHING
        # ====================================================

        print()
        print("STEP 2: SERVICE MATCHING")

        matched_services = match_services(
            original_requirement=customer_requirement,
            analysis=analysis,
            services=self.services,
            embed_model=self.embed_model,
        )

        print(
            f"Selected {len(matched_services)} service(s)."
        )

        for service in matched_services:

            print(
                f"- {service['name']} "
                f"({service['service_id']}) "
                f"keyword={service['keyword_score']:.4f} "
                f"semantic={service['semantic_score']:.4f} "
                f"final={service['score']:.4f}"
            )

        # ====================================================
        # NO MATCH
        # ====================================================

        if not matched_services:

            print()
            print(
                "No suitable MPI service was identified."
            )

            return build_response(
                original_requirement=customer_requirement,
                matched_services=[],
                evidence=[],
            )

        # ====================================================
        # STEP 3: MPI KNOWLEDGE RETRIEVAL
        # ====================================================

        print()
        print("STEP 3: MPI KNOWLEDGE RETRIEVAL")

        evidence = retrieve_evidence(
            requirement=customer_requirement,
            matched_services=matched_services,
            embed_model=self.embed_model,
        )

        print(
            f"Retrieved {len(evidence)} "
            "verified evidence item(s)."
        )

        # ====================================================
        # STEP 4: RESPONSE GENERATION
        # ====================================================

        print()
        print("STEP 4: RESPONSE GENERATION")

        response = build_response(
            original_requirement=customer_requirement,
            matched_services=matched_services,
            evidence=evidence,
        )

        print()
        print("=" * 60)
        print(response)
        print("=" * 60)

        return response