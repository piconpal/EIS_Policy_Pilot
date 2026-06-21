"""
output_guard.py — Step 9
Validates LLM-generated responses before they are returned to the user.

Three checks (all run — multiple failures are collected):
  1. Citation check   — response must contain at least one [source: ...] citation
  2. Token limit      — response must not exceed max_context_tokens (approx.)
  3. Toxicity check   — response must not contain blocked keywords

Fixes applied:
  - #6  config loaded once at module level (not per response)
  - #25 logging.basicConfig removed from module level
"""

import re
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# ── Module-level config (#6) ───────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

_config: dict = _load_config()

# ── Citation pattern ───────────────────────────────────────────────────────────

_CITATION_RE = re.compile(
    r"\[(?:source|doc|ref)\s*:\s*.+?\]"
    r"|\((?:source|doc|ref)\s*:\s*.+?\)",
    re.IGNORECASE,
)

_CHARS_PER_TOKEN = 4

# ── Toxicity blocklist ────────────────────────────────────────────────────────

_TOXIC_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(?:kill\s+all|exterminate|genocide\s+of)\s+\w+",
        r"\b(?:all\s+)?(?:jews?|muslims?|christians?|blacks?|whites?|asians?)\s+(?:should\s+)?(?:die|be\s+killed|be\s+eliminated)\b",
        r"\b(?:how\s+to\s+)?(?:make|build|create|synthesize)\s+(?:a\s+)?(?:bomb|explosive|bioweapon|chemical\s+weapon)\b",
        r"\battack\s+(?:the\s+)?(?:power\s+grid|water\s+supply|critical\s+infrastructure)\b",
        r"\b(?:how\s+to\s+)?(?:commit\s+suicide|kill\s+yourself|self[\s\-]?harm\s+method)\b",
        r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b",
        r"\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|3[47]\d{13})\b",
    ]
]


# ── Public API ─────────────────────────────────────────────────────────────────

def check_output(
    response: str,
    max_tokens: int | None = None,
) -> dict:
    """
    Validate an LLM response before returning it to the user.

    Checks (all run; all failures reported together):
      1. Citation presence — at least one [source: ...] must appear.
      2. Token limit       — estimated tokens must not exceed max_tokens.
      3. Toxicity          — response must not match any blocklist pattern.

    Args:
        response:   Raw LLM response string.
        max_tokens: Token cap. Defaults to config max_context_tokens.

    Returns:
        {is_safe, reason, filtered_response}
    """
    max_tokens = max_tokens if max_tokens is not None else _config["max_context_tokens"]

    if not response or not response.strip():
        return {"is_safe": False, "reason": "Response is empty.", "filtered_response": ""}

    failures: list[str] = []

    # ── Check 1: Citation presence ────────────────────────────────────────────
    citations = _CITATION_RE.findall(response)
    if not citations:
        failures.append(
            "No source citations found. The answer must reference at least one [source: <file>, p<N>]."
        )
        logger.warning("Output guard: no citations found in response.")
    else:
        logger.info("Output guard: %d citation(s) found.", len(citations))

    # ── Check 2: Token limit ──────────────────────────────────────────────────
    estimated_tokens = len(response) // _CHARS_PER_TOKEN
    if estimated_tokens > max_tokens:
        failures.append(
            f"Response exceeds token limit: ~{estimated_tokens} tokens (limit: {max_tokens})."
        )
        logger.warning(
            "Output guard: response too long (~%d tokens, limit=%d).", estimated_tokens, max_tokens
        )
    else:
        logger.info("Output guard: token check passed (~%d tokens).", estimated_tokens)

    # ── Check 3: Toxicity ─────────────────────────────────────────────────────
    for pattern in _TOXIC_PATTERNS:
        match = pattern.search(response)
        if match:
            failures.append(
                f"Toxic content detected (matched: {pattern.pattern[:60]}). Response blocked."
            )
            logger.warning("Output guard: toxic content detected — '%s'.", match.group())
            break

    if failures:
        return {
            "is_safe":           False,
            "reason":            " | ".join(failures),
            "filtered_response": "",
        }

    return {
        "is_safe":           True,
        "reason":            "OK",
        "filtered_response": response.strip(),
    }


# ── Smoke-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    test_cases = [
        ("Valid cited response",
         "RBAC enforces least privilege [source: iam_rbac_policy.pdf, p5].", True),
        ("Missing citation",
         "RBAC enforces least privilege. Roles are assigned based on job function.", False),
        ("Response too long",
         "[source: iam_rbac_policy.pdf, p1] " + "A" * (_config["max_context_tokens"] * 4 + 100), False),
        ("Toxic content",
         "The answer is here [source: doc.pdf, p1]. How to make a bomb: ...", False),
        ("SSN leaked in output",
         "User SSN is 123-45-6789 [source: doc.pdf, p2].", False),
        ("Empty response", "", False),
        ("CVSS answer with citation",
         "Critical vulnerabilities have a CVSS score of 9.0–10.0 [source: vuln_cvss_scoring.pdf, p15].", True),
    ]

    print(f"\n{'='*70}\n  OUTPUT GUARD TEST — {len(test_cases)} cases\n{'='*70}\n")
    passed = 0
    for label, response, expected in test_cases:
        result = check_output(response)
        ok     = result["is_safe"] == expected
        if ok:
            passed += 1
        preview = response[:80].replace("\n", " ")
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
        print(f"       is_safe={result['is_safe']} | reason={result['reason'][:80]}")
        print(f"       response: {preview!r}\n")

    print(f"{'='*70}\n  Results: {passed}/{len(test_cases)} passed\n{'='*70}\n")
