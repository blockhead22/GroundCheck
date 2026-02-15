"""GroundCheck CLI — verify text against memories from the command line.

Usage:
    groundcheck verify "You work at Amazon" --memories memories.json
    groundcheck extract "My name is Alice and I work at Google"
    groundcheck version
"""

import argparse
import json
import sys
import time
from typing import List

from . import __version__
from .types import Memory
from .verifier import GroundCheck
from .fact_extractor import extract_fact_slots


def _load_memories(path: str) -> List[Memory]:
    """Load memories from a JSON file.

    Accepts either a list of objects or a dict with a ``memories`` key.
    Each object must have at least ``text``; ``id``, ``trust``, and
    ``source`` are optional.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = data.get("memories", data.get("facts", []))

    memories: List[Memory] = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            memories.append(Memory(id=f"m{i}", text=item))
        elif isinstance(item, dict):
            memories.append(
                Memory(
                    id=item.get("id", f"m{i}"),
                    text=item["text"],
                    trust=item.get("trust", 1.0),
                    source=item.get("source", "document"),
                )
            )
        else:
            print(f"Warning: skipping unrecognised memory entry at index {i}", file=sys.stderr)
    return memories


def cmd_verify(args: argparse.Namespace) -> int:
    """Run verification and print the report."""
    verifier = GroundCheck(neural=args.neural)

    if not args.memories:
        print("Error: --memories is required for verify", file=sys.stderr)
        return 1

    memories = _load_memories(args.memories)
    if not memories:
        print("Warning: no memories loaded from file", file=sys.stderr)

    t0 = time.perf_counter()
    report = verifier.verify(args.text, memories, mode=args.mode)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    output = {
        "passed": report.passed,
        "confidence": round(report.confidence, 4),
        "hallucinations": report.hallucinations,
        "corrected": report.corrected,
        "facts_extracted": {
            k: {"slot": v.slot, "value": v.value}
            for k, v in report.facts_extracted.items()
        },
        "facts_supported": {
            k: {"slot": v.slot, "value": v.value}
            for k, v in report.facts_supported.items()
        },
        "contradictions": [
            {
                "slot": c.slot,
                "values": c.values,
                "most_trusted_value": c.most_trusted_value,
            }
            for c in report.contradiction_details
        ],
        "latency_ms": round(elapsed_ms, 2),
        "memories_count": len(memories),
    }

    print(json.dumps(output, indent=2))
    return 0 if report.passed else 1


def cmd_extract(args: argparse.Namespace) -> int:
    """Extract facts from text and print them."""
    facts = extract_fact_slots(args.text)
    output = {
        slot: {"value": f.value, "normalized": f.normalized, "slot": f.slot}
        for slot, f in facts.items()
    }
    print(json.dumps(output, indent=2))
    return 0


def cmd_version(_args: argparse.Namespace) -> int:
    """Print the version."""
    print(f"groundcheck {__version__}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="groundcheck",
        description="GroundCheck — trust-weighted hallucination detection for AI agents",
    )
    sub = parser.add_subparsers(dest="command")

    # groundcheck verify
    p_verify = sub.add_parser("verify", help="Verify text against stored memories")
    p_verify.add_argument("text", help="The text to verify")
    p_verify.add_argument(
        "--memories", "-m",
        required=True,
        help="Path to a JSON file containing memories (list of {text, trust?, source?})",
    )
    p_verify.add_argument(
        "--mode",
        choices=["strict", "permissive"],
        default="strict",
        help="Verification mode (default: strict)",
    )
    p_verify.add_argument(
        "--neural",
        action="store_true",
        default=False,
        help="Enable neural/semantic matching (requires groundcheck[neural])",
    )
    p_verify.set_defaults(func=cmd_verify)

    # groundcheck extract
    p_extract = sub.add_parser("extract", help="Extract facts from text")
    p_extract.add_argument("text", help="The text to extract facts from")
    p_extract.set_defaults(func=cmd_extract)

    # groundcheck version
    p_version = sub.add_parser("version", help="Show version")
    p_version.set_defaults(func=cmd_version)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
