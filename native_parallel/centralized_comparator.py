"""Centralized comparator adapter for native BFCL tool calling."""

from typing import Any, Dict, Sequence

from comlrl.trainers.centralized import (
    CentralizedComparatorParseError,
    TaggedCentralizedComparatorAdapter,
)


class BFCLCentralizedComparatorAdapter:
    def __init__(self) -> None:
        self._tagged_parser = TaggedCentralizedComparatorAdapter()

    def build_prompt(
        self,
        batch_item: Dict[str, Any],
        agent_prompts: Sequence[str],
    ) -> str:
        del batch_item
        prompt_sections = "\n\n".join(
            f"Agent {agent_idx} original prompt:\n{prompt}"
            for agent_idx, prompt in enumerate(agent_prompts)
        )
        output_sections = "\n".join(
            f"<agent_{agent_idx}>\nfunction calls for Agent {agent_idx}, or []\n"
            f"</agent_{agent_idx}>"
            for agent_idx in range(len(agent_prompts))
        )
        return f"""You are acting as one centralized coordinator for {len(agent_prompts)} function-calling agents.

Generate the exact separate output each decentralized agent would submit. Respect
each agent's assignment from its original prompt. Across all outputs, cover the
requested tool-call intents without unnecessary duplication. Inside each section,
use only the Python-style function calls required by that agent, one call per line,
or [] when that agent has no valid call. Do not add explanations or markdown.

{prompt_sections}

Return exactly this structure:
{output_sections}
"""

    def parse_completion(
        self,
        completion: str,
        batch_item: Dict[str, Any],
        num_agents: int,
    ) -> Sequence[str]:
        try:
            return self._tagged_parser.parse_completion(
                completion,
                batch_item,
                num_agents,
            )
        except CentralizedComparatorParseError:
            return [""] * num_agents
