"""GroundCheck MCP Server — exposes store/check/verify tools via Model Context Protocol.

Usage:
    groundcheck-mcp --db .groundcheck/memory.db
    
    # Or via Python:
    python -m groundcheck_mcp.server --db .groundcheck/memory.db
"""

import argparse
import json
import logging
import sys
from typing import Optional

from mcp.server import FastMCP

from groundcheck import GroundCheck, VerificationReport
from groundcheck.fact_extractor import extract_fact_slots
from .storage import MemoryStore

logger = logging.getLogger("groundcheck-mcp")

# Global state
_store: Optional[MemoryStore] = None
_verifier: Optional[GroundCheck] = None

# Create the FastMCP server
mcp = FastMCP(
    "groundcheck",
    instructions=(
        "GroundCheck provides trust-weighted memory and hallucination detection. "
        "Use crt_store_fact when the user states facts. "
        "Use crt_check_memory before answering questions about the user/project. "
        "Use crt_verify_output before sending responses that reference stored facts."
    ),
)


def _get_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore(":memory:")
    return _store


def _get_verifier() -> GroundCheck:
    global _verifier
    if _verifier is None:
        _verifier = GroundCheck()
    return _verifier


def _report_to_dict(report: VerificationReport) -> dict:
    """Convert a VerificationReport to a JSON-serializable dict."""
    return {
        "passed": report.passed,
        "corrected": report.corrected,
        "hallucinations": report.hallucinations,
        "confidence": report.confidence,
        "facts_extracted": {
            k: {"slot": v.slot, "value": v.value, "normalized": v.normalized}
            for k, v in report.facts_extracted.items()
        },
        "facts_supported": {
            k: {"slot": v.slot, "value": v.value, "normalized": v.normalized}
            for k, v in report.facts_supported.items()
        },
        "contradicted_claims": report.contradicted_claims,
        "contradiction_details": [
            {
                "slot": c.slot,
                "values": c.values,
                "memory_ids": c.memory_ids,
                "most_trusted_value": c.most_trusted_value,
                "most_recent_value": c.most_recent_value,
            }
            for c in report.contradiction_details
        ],
        "requires_disclosure": report.requires_disclosure,
        "expected_disclosure": report.expected_disclosure,
    }


@mcp.tool()
def crt_store_fact(
    text: str,
    source: str = "user",
    thread_id: str = "default",
) -> str:
    """Store a user fact into persistent memory with contradiction detection.
    
    Call this whenever the user states something about themselves,
    their project, preferences, or environment. Returns the stored 
    memory and any contradictions detected against existing memories.
    
    Args:
        text: The fact to store (e.g. 'User works at Microsoft')
        source: Source of the fact — user|document|code|inferred. Affects trust score.
        thread_id: Thread/conversation ID for memory isolation.
    """
    store = _get_store()
    verifier = _get_verifier()

    # Store the new memory
    new_mem = store.store(text=text, thread_id=thread_id, source=source)

    # Get all memories for contradiction check
    all_memories = store.get_all(thread_id=thread_id)
    
    # Extract facts from the new text
    new_facts = extract_fact_slots(text)
    
    # Check for contradictions against existing memories (excluding just-stored)
    existing = [m for m in all_memories if m.id != new_mem.id]
    
    contradictions = []
    if existing and new_facts:
        report = verifier.verify(text, existing, mode="permissive")
        if report.contradiction_details:
            contradictions = [
                {
                    "slot": c.slot,
                    "values": c.values,
                    "most_trusted_value": c.most_trusted_value,
                    "most_recent_value": c.most_recent_value,
                    "action": "Ask user to confirm which is current",
                }
                for c in report.contradiction_details
            ]

    result = {
        "stored": True,
        "memory_id": new_mem.id,
        "text": new_mem.text,
        "trust": new_mem.trust,
        "source": source,
        "facts_extracted": {k: v.value for k, v in new_facts.items()} if new_facts else {},
        "total_memories": len(all_memories),
        "contradictions": contradictions,
        "has_contradiction": len(contradictions) > 0,
    }

    return json.dumps(result, indent=2)


@mcp.tool()
def crt_check_memory(
    query: str,
    thread_id: str = "default",
) -> str:
    """Query stored facts before answering. Returns memories with trust scores and contradiction warnings.
    
    Call this before answering questions about the user, their project,
    or their preferences to ensure your response is grounded in stored facts.
    
    Args:
        query: What to search for in memory (e.g. 'database technology', 'employer')
        thread_id: Thread/conversation ID.
    """
    store = _get_store()
    verifier = _get_verifier()

    memories = store.query(query=query, thread_id=thread_id)

    if not memories:
        result = {
            "found": 0,
            "memories": [],
            "contradictions": [],
            "note": "No memories stored for this thread yet.",
        }
        return json.dumps(result, indent=2)

    # Check for internal contradictions among retrieved memories
    combined_text = " ".join(m.text for m in memories)
    report = verifier.verify(combined_text, memories, mode="permissive")

    result = {
        "found": len(memories),
        "memories": [
            {
                "id": m.id,
                "text": m.text,
                "trust": m.trust,
                "timestamp": m.timestamp,
            }
            for m in memories
        ],
        "contradictions": [
            {
                "slot": c.slot,
                "values": c.values,
                "most_trusted_value": c.most_trusted_value,
                "most_recent_value": c.most_recent_value,
            }
            for c in report.contradiction_details
        ],
    }

    return json.dumps(result, indent=2)


@mcp.tool()
def crt_verify_output(
    draft: str,
    thread_id: str = "default",
    mode: str = "strict",
) -> str:
    """Verify your draft response against stored memories before sending it.
    
    Returns hallucination list, corrected text, and confidence score.
    Call this before every response that references user facts.
    
    Args:
        draft: The draft response text to verify.
        thread_id: Thread/conversation ID.
        mode: 'strict' rewrites hallucinations; 'permissive' reports only.
    """
    store = _get_store()
    verifier = _get_verifier()

    memories = store.get_all(thread_id=thread_id)

    if not memories:
        result = {
            "passed": True,
            "note": "No memories to verify against. Passing by default.",
            "confidence": 0.0,
        }
        return json.dumps(result, indent=2)

    report = verifier.verify(draft, memories, mode=mode)
    result = _report_to_dict(report)

    return json.dumps(result, indent=2)


def main():
    """Entry point for the groundcheck-mcp command."""
    parser = argparse.ArgumentParser(description="GroundCheck MCP Server")
    parser.add_argument(
        "--db",
        default=".groundcheck/memory.db",
        help="Path to SQLite database for persistent memory (default: .groundcheck/memory.db)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Initialize global store with the specified DB path
    global _store
    _store = MemoryStore(args.db)
    logger.info(f"GroundCheck MCP server started with db={args.db}")

    # Run via stdio (standard for MCP)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
