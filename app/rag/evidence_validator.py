"""
MPI evidence validation and response construction.

Responsibilities:
- Keep the final response limited to services selected by
  the deterministic matcher.
- Report whether verified knowledge exists.
- Never invent pricing, eligibility, turnaround time,
  deliverables, capabilities, or funding information.
- Use actual repository source paths.
"""

from __future__ import annotations

from typing import Dict, List

from app.config.settings import PROJECT_ROOT
from app.matcher.service_matcher import (
    SERVICE_KEYWORDS,
    normalize_text,
)


def build_response(
    original_requirement: str,
    matched_services: List[Dict],
    evidence: List[Dict],
) -> str:
    """
    Build the customer-facing MPI response.

    This function is deliberately deterministic.
    The LLM is not used to select, rename, reorder, or
    fabricate services.
    """

    if not matched_services:

        return (
            "MPI RESPONSE\n\n"
            "Customer Requirement:\n"
            f"{original_requirement}\n\n"
            "Recommended MPI Services:\n"
            "No suitable MPI service was identified for "
            "this requirement from the current MPI service catalog."
        )

    evidence_by_filename: Dict[str, List[Dict]] = {}

    for item in evidence:

        evidence_by_filename.setdefault(
            item["filename"],
            [],
        ).append(item)

    lines = [
        "MPI RESPONSE",
        "",
        "Customer Requirement:",
        original_requirement,
        "",
        "Recommended MPI Services:",
        "",
    ]

    for index, service in enumerate(
        matched_services,
        start=1,
    ):

        lines.append(
            f"{index}. {service['name']}"
        )

        lines.append(
            f"   Service ID: {service['service_id']}"
        )

        keywords = SERVICE_KEYWORDS.get(
            service["filename"],
            [],
        )

        matched_keywords = [
            keyword
            for keyword in keywords
            if normalize_text(keyword)
            in normalize_text(
                original_requirement
            )
        ]

        if matched_keywords:

            # Remove duplicate keyword variants while
            # preserving their original order.
            unique_keywords = list(
                dict.fromkeys(
                    matched_keywords
                )
            )

            lines.append(
                "   Why it matches: Customer explicitly "
                "mentioned "
                f"{', '.join(unique_keywords[:3])}."
            )

        else:

            lines.append(
                "   Why it matches: The service matched "
                "the customer's stated requirements."
            )

        service_evidence = evidence_by_filename.get(
            service["filename"],
            [],
        )

        if service_evidence:

            lines.append(
                "   Verified MPI information: Available in "
                "the current MPI knowledge base."
            )

        else:

            lines.append(
                "   Verified MPI information: Not currently "
                "available in the MPI knowledge base."
            )

        lines.append("")

    # --------------------------------------------------------
    # INFORMATION AVAILABLE
    # --------------------------------------------------------

    lines.append("Information Available:")

    verified_services = []

    for service in matched_services:

        service_evidence = evidence_by_filename.get(
            service["filename"],
            [],
        )

        if service_evidence:
            verified_services.append(
                service["name"]
            )

    if verified_services:

        for service_name in verified_services:

            lines.append(
                f"- {service_name}: Verified source content "
                "is available."
            )

    else:

        lines.append(
            "- No detailed verified MPI service information "
            "is currently available for these services."
        )

    # --------------------------------------------------------
    # INFORMATION NOT AVAILABLE
    # --------------------------------------------------------

    lines.append("")
    lines.append("Information Not Available:")

    lines.append(
        "- Pricing, turnaround time, eligibility, "
        "deliverables and other operational details are "
        "not stated unless verified in the MPI knowledge base."
    )

    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    lines.append("")
    lines.append("Sources:")

    for service in matched_services:

        relative_path = service["path"].relative_to(
            PROJECT_ROOT
        )

        lines.append(
            f"- {relative_path}"
        )

    return "\n".join(lines)