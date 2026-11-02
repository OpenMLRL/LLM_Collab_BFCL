"""Native centralized BFCL prompt: one planner writes both agents' sections.

The concatenated adapter embeds every decentralized agent prompt, so the tool
schemas, the user request and the "you cannot communicate" instructions appear
once per agent. This prompt states the request and schemas once and asks one
planner for both tagged sections, which the unchanged parser splits back into
the agents' outputs for the same reward.
"""

from typing import Any, Dict, Sequence

from native_parallel.centralized_comparator import BFCLCentralizedComparatorAdapter
from native_parallel.formatting import _function_docs, _metadata_lines


def _section_assignment(role_mode: str) -> str:
    """Describe the two sections with the same split the agents' roles use."""

    if role_mode == "same_prompt":
        raise ValueError(
            "The native centralized prompt needs complementary sections; "
            "role_mode=same_prompt gives both agents every intent."
        )
    if role_mode == "self_select":
        return (
            "Divide the needed calls between the two sections yourself. Every needed\n"
            "call appears in exactly one section, and when two or more calls are\n"
            "needed, each section contains at least one of them."
        )
    if role_mode == "odd_even":
        return (
            "Section agent_0 contains the odd-numbered tool-call intents and section\n"
            "agent_1 the even-numbered ones, using 1-based ordering from left to right\n"
            "in the user's request."
        )
    return (
        "Section agent_0 contains the first half of the tool-call intents in\n"
        "user-request order, taking the extra middle intent when the count is odd;\n"
        "section agent_1 contains the second half."
    )


def build_native_centralized_prompt(
    example: Dict[str, Any], *, role_mode: str = "self_select"
) -> str:
    user_prompt = example.get("user_prompt") or example.get("prompt") or ""
    return f"""You are one function-calling planner. The user's request is split into two
parts, and you write both parts yourself as two separate sections, agent_0 and
agent_1. A downstream aggregator merges the calls from both sections and scores
them as one answer.

Assignment:
{_section_assignment(role_mode)}
If the request needs no valid call, write [] in both sections.

{_metadata_lines(example)}

Available function schemas:
{_function_docs(example)}

User request:
{user_prompt}

Output requirements:
- Inside each section, use Python-style function-call syntax, one call per line.
- Use keyword arguments from the function schemas.
- Do not include explanations, markdown, numbering, or reasoning.
- A section with no call contains exactly: []

Example call format:
function_name(arg1="value", arg2=3)
another.function_name(flag=True)

Return exactly this structure:
<agent_0>
calls for section agent_0, or []
</agent_0>
<agent_1>
calls for section agent_1, or []
</agent_1>
"""


class BFCLNativeCentralizedAdapter(BFCLCentralizedComparatorAdapter):
    """State the request once; parse the same tagged sections."""

    def __init__(self, role_mode: str = "self_select") -> None:
        super().__init__()
        _section_assignment(role_mode)  # Reject unsupported role modes up front.
        self.role_mode = role_mode

    def build_prompt(
        self,
        batch_item: Dict[str, Any],
        agent_prompts: Sequence[str],
    ) -> str:
        if len(agent_prompts) != 2:
            raise ValueError("The native BFCL centralized prompt has two sections.")
        return build_native_centralized_prompt(batch_item, role_mode=self.role_mode)
