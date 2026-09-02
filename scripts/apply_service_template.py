from pathlib import Path
import re


PRODUCTS_DIR = Path("products")


def title_from_filename(filename: str) -> str:
    """
    Converts:
        03_gst_filing.md
    into:
        GST Filing
    """
    name = Path(filename).stem
    name = re.sub(r"^\d+_", "", name)
    return name.replace("_", " ").title()


def get_category(path: Path) -> str:
    """
    Converts:
        04_compliance_and_legal_support
    into:
        Compliance And Legal Support
    """
    category = path.parent.name
    category = re.sub(r"^\d+_", "", category)
    return category.replace("_", " ").title()


def get_category_code(path: Path) -> str:
    """
    Uses the category number:
        04_compliance_and_legal_support
    -> COMP
    """
    category_name = re.sub(r"^\d+_", "", path.parent.name)

    mapping = {
        "packaging_and_printing": "PACK",
        "prototyping_and_product_development": "PROTO",
        "it_and_digital_services": "IT",
        "compliance_and_legal_support": "COMP",
        "logistics_and_operations": "LOG",
        "marketing_and_sales_support": "MKT",
        "business_and_finance_services": "FIN",
        "specialized_startup_support": "STARTUP",
    }

    return mapping.get(category_name, "MPI")


def get_service_number(path: Path) -> str:
    match = re.match(r"^(\d+)_", path.name)

    if match:
        return match.group(1).zfill(3)

    return "000"


def create_template(path: Path) -> str:
    service_name = title_from_filename(path.name)
    category = get_category(path)
    code = get_category_code(path)
    service_number = get_service_number(path)

    service_id = f"MPI-{code}-{service_number}"

    return f"""# {service_name}

## Service ID
{service_id}

## Category
{category}

## Description
[ADD VERIFIED MPI INFORMATION]

## Customer Problems Solved
[ADD VERIFIED INFORMATION]

## Suitable Customers
[ADD VERIFIED CUSTOMER SEGMENTS]

## Customer Requirements
[ADD VERIFIED INFORMATION/DOCUMENTS REQUIRED]

## Deliverables
[ADD VERIFIED INFORMATION]

## Use Cases
[ADD VERIFIED INFORMATION]

## Process
[ADD VERIFIED PROCESS]

## Pricing
[ADD VERIFIED PRICING]

## Turnaround Time
[ADD VERIFIED TURNAROUND TIME]

## Eligibility
[ADD VERIFIED ELIGIBILITY INFORMATION]

## Limitations
[ADD VERIFIED LIMITATIONS]

## Frequently Asked Questions

### FAQ 1
[ADD VERIFIED ANSWER]

### FAQ 2
[ADD VERIFIED ANSWER]

### FAQ 3
[ADD VERIFIED ANSWER]

## Related Services
[ADD VERIFIED RELATED MPI SERVICES]

## Keywords
[ADD SEARCH KEYWORDS]

## Source
MPI Products and Services master list
"""


def main():
    files = sorted(PRODUCTS_DIR.rglob("*.md"))

    if not files:
        print("No product files found.")
        return

    updated = 0

    for file_path in files:
        content = create_template(file_path)
        file_path.write_text(content, encoding="utf-8")
        updated += 1

    print(f"Updated {updated} service files.")


if __name__ == "__main__":
    main()