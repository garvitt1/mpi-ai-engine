"""
MPI customer requirement analyzer.
"""

from llama_index.llms.ollama import Ollama


def analyze_requirement(
    requirement: str,
    llm: Ollama,
) -> str:
    """
    Analyze a customer requirement without recommending services.
    """

    prompt = f"""
You are the MPI Requirement Analyzer.

Analyze the customer request.

Return exactly:

CUSTOMER_TYPE:
<value>

REQUIREMENTS:
- <requirement 1>
- <requirement 2>

CONSTRAINTS:
- <value or UNKNOWN>

IMPORTANT_DETAILS:
- <value or UNKNOWN>

UNKNOWN:
- <missing information>

CUSTOMER_MESSAGE:
<original message>

Rules:
- Extract only information explicitly stated or directly implied.
- "my startup" means CUSTOMER_TYPE can be STARTUP.
- Do not recommend MPI services.
- Do not invent MPI information.
- Do not invent pricing.
- Do not infer unsupported business facts.
- Do not turn missing information into facts.

Customer request:
{requirement}
""".strip()

    result = llm.complete(prompt)

    return result.text.strip()