"""
MPI AI Engine Core Tests.

These tests protect the deterministic service-matching behavior.
They do not require Ollama or Chroma.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.matcher.service_matcher import (  # noqa: E402
    filename_to_title,
    keyword_score,
    make_service_id,
    normalize_text,
)


def make_service(filename: str):
    """Create a minimal service object for matcher tests."""

    path = (
        PROJECT_ROOT
        / "products"
        / "03_it_and_digital_services"
        / filename
    )

    return {
        "path": path,
        "filename": filename,
        "name": filename_to_title(filename),
        "service_id": make_service_id(path),
        "category": "It And Digital Services",
        "content": "",
    }


def test_normalize_text():

    text = "GST Compliance Support!"

    assert normalize_text(text) == (
        "gst compliance support"
    )


def test_website_keyword_match():

    service = make_service(
        "01_website_development.md"
    )

    score = keyword_score(
        "I need a website for my startup.",
        service,
    )

    assert score > 0


def test_website_does_not_match_gst():

    service = make_service(
        "01_website_development.md"
    )

    score = keyword_score(
        "I need GST compliance support.",
        service,
    )

    assert score == 0


def test_website_service_id():

    service = make_service(
        "01_website_development.md"
    )

    assert service["service_id"] == "MPI-IT-001"


def test_gst_service_id():

    path = (
        PROJECT_ROOT
        / "products"
        / "04_compliance_and_legal_support"
        / "03_gst_filing.md"
    )

    service_id = make_service_id(path)

    assert service_id == "MPI-COMP-003"


def test_branding_keyword_match():

    path = (
        PROJECT_ROOT
        / "products"
        / "06_marketing_and_sales_support"
        / "01_brand_identity_design.md"
    )

    service = {
        "path": path,
        "filename": "01_brand_identity_design.md",
        "name": filename_to_title(
            "01_brand_identity_design.md"
        ),
        "service_id": make_service_id(path),
        "category": "Marketing And Sales Support",
        "content": "",
    }

    score = keyword_score(
        "I need branding for my startup.",
        service,
    )

    assert score > 0


def test_loan_has_no_known_website_keyword():

    service = make_service(
        "01_website_development.md"
    )

    score = keyword_score(
        "I need MPI to provide me with a ₹50,000 loan.",
        service,
    )

    assert score == 0