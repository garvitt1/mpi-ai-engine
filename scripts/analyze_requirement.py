from typing import Any, Dict

from llama_index.llms.ollama import Ollama


# ============================================================
# MPI CUSTOMER REQUIREMENT ANALYZER
# ============================================================

OLLAMA_MODEL = "llama3.2:3b"


# ============================================================
# Local LLM
# ============================================================

llm = Ollama(
    model=OLLAMA_MODEL,
    request_timeout=120.0,
)


# ============================================================
# Prompt
# ============================================================

ANALYSIS_PROMPT = """
You are the customer requirement analyzer for MPI.

Your job is to extract ONLY information explicitly stated
or directly expressed in the customer message.

Do not invent information.

Do not recommend MPI services.

Do not provide prices.

Do not provide eligibility.

Do not infer a specific budget when no amount is given.

Do not assume a customer type unless it is stated or clearly
expressed.

Return the result in exactly this format:

CUSTOMER_TYPE:
[customer type or UNKNOWN]

REQUIREMENTS:
- [requirement 1]
- [requirement 2]

CONSTRAINTS:
- [constraint 1]
- [constraint 2]

IMPORTANT_DETAILS:
- [important detail 1]
- [important detail 2]

UNKNOWN:
- [important information that was not provided]

CUSTOMER_MESSAGE:
[original message]

Customer message:
"""


# ============================================================
# Analyze requirement
# ============================================================

def analyze_requirement(
    customer_message: str,
) -> Dict[str, Any]:

    message = customer_message.strip()

    if not message:
        raise ValueError(
            "Customer message cannot be empty."
        )

    prompt = (
        ANALYSIS_PROMPT
        + "\n"
        + message
    )

    try:

        response = llm.complete(
            prompt
        )

    except Exception as error:

        raise RuntimeError(
            f"Customer requirement analysis failed: "
            f"{error}"
        ) from error

    result = response.text.strip()

    if not result:
        raise RuntimeError(
            "The local LLM returned an empty analysis."
        )

    return {
        "analysis": result,
        "original_message": message,
    }


# ============================================================
# Interactive test mode
# ============================================================

def main():

    print("\n======================================")
    print("MPI CUSTOMER REQUIREMENT ANALYZER")
    print("======================================")
    print("Describe the customer's requirement.")
    print("Type 'exit' to stop.\n")

    while True:

        try:

            customer_message = input(
                "Customer requirement: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print(
                "\nExiting analyzer."
            )

            break

        if customer_message.lower() == "exit":

            print(
                "Exiting analyzer."
            )

            break

        if not customer_message:

            print(
                "Please enter a customer requirement.\n"
            )

            continue

        try:

            result = analyze_requirement(
                customer_message
            )

        except (
            ValueError,
            RuntimeError,
        ) as error:

            print(
                f"\nMPI ERROR: {error}\n"
            )

            continue

        print(
            "\n======================================"
        )

        print(
            "REQUIREMENT ANALYSIS"
        )

        print(
            "======================================"
        )

        print(
            result["analysis"]
        )

        print()


if __name__ == "__main__":
    main()