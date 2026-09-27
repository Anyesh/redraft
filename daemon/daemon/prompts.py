"""The section prompt. Bump TEMPLATE_VERSION on any change to the rendered text,
because Loom records it per turn and measurements compare like with like."""

from collections.abc import Sequence
from typing import Literal

from daemon.sessions import Source

TEMPLATE_VERSION = 1

Kind = Literal["rederive", "revise"]

REVISE_NOTE = (
    "The current draft below was edited by hand. Rewrite the draft so all of it is "
    "consistent with the sources and with those edits: keep the edited content, drop "
    "lines the edits make redundant, and change the rest only where the edits make it "
    "inconsistent. Output only the complete revised draft, with no commentary."
)


def render(
    kind: Kind,
    instruction: str,
    sources: Sequence[Source],
    draft: str,
    pinned: Sequence[str],
) -> tuple[str, str]:
    """Return (context, question) for the engine's chat template.

    Sources come first and the draft last, so a draft-only edit keeps the whole
    source prefix cached on an affine slot.
    """
    blocks = [f"## {s.name}\n{s.text}" for s in sources]
    if pinned:
        lines = "\n".join(pinned)
        blocks.append(f"## Keep these lines exactly as written\n{lines}")
    question = instruction
    if kind == "revise":
        blocks.append(f"## Current draft\n{draft}")
        question = f"{instruction}\n\n{REVISE_NOTE}"
    return "\n\n".join(blocks), question
