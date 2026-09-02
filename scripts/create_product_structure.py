from pathlib import Path


PRODUCTS = {
    "01_packaging_and_printing": [
        "custom_cartons",
        "corrugated_boxes",
        "folding_cartons",
        "rigid_boxes",
        "product_labels",
        "barcode_stickers",
        "tamper_evident_seals",
        "pouches_and_sachets",
        "blister_packaging",
        "shrink_wrapping",
        "stretch_film",
        "bubble_wrap",
        "thermal_tags",
        "instruction_leaflets",
        "brochures_and_catalogs",
    ],

    "02_prototyping_and_product_development": [
        "3d_printing",
        "rapid_prototyping",
        "cad_design",
        "industrial_design",
        "model_making",
        "electronics_prototyping",
        "mechanical_design",
        "testing_samples",
    ],

    "03_it_and_digital_services": [
        "website_development",
        "mobile_app_development",
        "ui_ux_design",
        "erp_setup",
        "crm_setup",
        "software_qa_testing",
        "cloud_hosting_setup",
        "cybersecurity_audit",
        "api_integration",
        "data_entry_and_digitization",
    ],

    "04_compliance_and_legal_support": [
        "company_incorporation_support",
        "udyam_registration_support",
        "gst_filing",
        "tds_filing",
        "roc_compliance",
        "trademark_filing",
        "copyright_filing",
        "patent_drafting_support",
        "iso_certification_support",
        "zed_certification_support",
    ],

    "05_logistics_and_operations": [
        "local_transportation",
        "courier_services",
        "warehouse_rental",
        "packaging_fulfillment",
        "last_mile_delivery",
        "inventory_management",
        "order_tracking_setup",
        "reverse_logistics",
        "loading_and_unloading",
        "cold_chain_handling",
    ],

    "06_marketing_and_sales_support": [
        "brand_identity_design",
        "logo_design",
        "social_media_management",
        "performance_marketing",
        "product_photography",
        "video_editing",
        "catalog_design",
        "sales_deck_design",
        "market_research",
        "lead_generation",
    ],

    "07_business_and_finance_services": [
        "accounting",
        "bookkeeping",
        "mis_reporting",
        "cma_preparation",
        "project_report_preparation",
        "valuation_support",
        "due_diligence_support",
        "business_analysis",
        "payroll_processing",
        "virtual_cfo_services",
    ],

    "08_specialized_startup_support": [
        "lab_testing",
        "quality_assurance",
        "certification_testing",
        "sample_sourcing",
        "toolroom_support",
        "packaging_design_consultation",
        "procurement_advisory",
        "vendor_onboarding",
        "b2b_sourcing_coordination",
        "custom_small_batch_manufacturing",
    ],
}


def display_name(slug: str) -> str:
    return slug.replace("_", " ").title()


def create_structure():
    base = Path("products")

    total = 0

    for category, services in PRODUCTS.items():
        category_path = base / category
        category_path.mkdir(parents=True, exist_ok=True)

        for number, service in enumerate(services, start=1):
            filename = f"{number:02d}_{service}.md"
            filepath = category_path / filename

            title = display_name(service)
            category_title = display_name(category[3:])

            content = f"""# {title}

## Service Category
{category_title}

## Service ID
MPI-{category[:2].upper()}-{number:03d}

## Description
[Add verified MPI description]

## Customer Problem
[Add the customer problem this service solves]

## Suitable Customers
[Add verified customer segments]

## Customer Requirements
[Add required information/documents]

## Deliverables
[Add verified deliverables]

## Use Cases
[Add verified use cases]

## Process
[Add verified process]

## Pricing
[Add verified pricing]

## Turnaround Time
[Add verified turnaround time]

## Eligibility
[Add verified eligibility information]

## Limitations
[Add verified limitations]

## FAQs
[Add verified FAQs]

## Related Services
[Add related MPI services]

## Source
MPI Products and Services master list
"""

            filepath.write_text(content)
            total += 1

    print(f"Created {total} service files.")


if __name__ == "__main__":
    create_structure()