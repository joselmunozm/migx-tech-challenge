"""Check that citation labels point at excerpts that were actually retrieved."""

import re

from app.prompts import ABSTAIN_RESPONSE

REF_PATTERN = re.compile(r"\[(S\d+)\]")


def check_citations(answer: str, allowed_refs: set[str]) -> dict:
    cited = sorted(set(REF_PATTERN.findall(answer)))
    unknown = sorted(set(cited) - allowed_refs)
    abstained = ABSTAIN_RESPONSE.lower() in answer.lower()
    grounded = not unknown and (bool(cited) or abstained)
    return {
        "ok": grounded,
        "cited": cited,
        "unknown_refs": unknown,
        "abstained": abstained,
    }
