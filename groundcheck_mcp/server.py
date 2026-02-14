"""GroundCheck MCP Server — exposes store/check/verify tools via Model Context Protocol.

Usage:
    groundcheck-mcp --db .groundcheck/memory.db
    groundcheck-mcp --db .groundcheck/memory.db --namespace my-project
    
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
_default_namespace: str = "default"

# Create the FastMCP server
mcp = FastMCP(
    "groundcheck",
    instructions=(
        "GroundCheck provides trust-weighted memory and hallucination detection. "
        "ALWAYS call crt_check_memory at the start of every turn, passing the "
        "user's latest message as the 'context' parameter — this automatically "
        "extracts and stores facts without needing a separate crt_store_fact call. "
        "Use crt_store_fact only for explicit corrections or important facts the "
        "auto-extractor might miss. "
        "Use crt_verify_output before sending responses that reference stored facts. "
        "Use namespace='global' for personal user facts (name, preferences) that "
        "should be available across all projects."
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
    namespace: str = "",
) -> str:
    """Store a user fact into persistent memory with contradiction detection.
    
    Call this whenever the user states something about themselves,
    their project, preferences, or environment. Returns the stored 
    memory and any contradictions detected against existing memories.
    
    Args:
        text: The fact to store (e.g. 'User works at Microsoft')
        source: Source of the fact — user|document|code|inferred. Affects trust score.
        thread_id: Thread/conversation ID for memory isolation.
        namespace: Project scope. Use 'global' for personal user facts
            (name, preferences) that should be available in every project.
            Leave empty to use the server's default namespace.
    """
    ns = namespace or _default_namespace
    store = _get_store()
    verifier = _get_verifier()

    # Store the new memory
    new_mem = store.store(text=text, thread_id=thread_id, source=source, namespace=ns)

    # Get all memories for contradiction check (INCLUDING the new one)
    all_memories = store.get_all(thread_id=thread_id, namespace=ns)
    
    # Extract facts from the new text
    new_facts = extract_fact_slots(text)
    
    # Detect contradictions across ALL memories (the verifier compares memories
    # against each other on mutually exclusive slots)
    contradictions = []
    if len(all_memories) > 1 and new_facts:
        # Verify combined text against all memories — this triggers
        # _detect_contradictions which compares memories cross-wise
        combined = " ".join(m.text for m in all_memories)
        report = verifier.verify(combined, all_memories, mode="permissive")
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
        "namespace": ns,
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
    namespace: str = "",
    include_global: bool = True,
    context: str = "",
) -> str:
    """Query stored facts and passively learn from conversation context.
    
    Call this at the START of every turn. Pass the user's latest message
    as 'context' — any facts in it will be automatically extracted and
    stored. You do NOT need to separately call crt_store_fact for facts
    that appear in the user's message.
    
    Returns existing memories with trust scores and contradiction warnings.
    By default, includes memories from the 'global' namespace so personal
    user facts (name, preferences) are always available.
    
    Args:
        query: What to search for in memory (e.g. 'database technology', 'employer')
        thread_id: Thread/conversation ID.
        namespace: Project scope to query. Leave empty for server default.
        include_global: Whether to also return 'global' namespace memories (default True).
        context: The user's latest message. Facts are silently extracted and stored.
    """
    ns = namespace or _default_namespace
    store = _get_store()
    verifier = _get_verifier()

    # ── Auto-learning: silently extract and store facts from context ──
    auto_learned = {}
    if context and context.strip():
        extracted = extract_fact_slots(context)
        if extracted:
            # Check existing memories to avoid storing duplicates
            existing = store.get_all(thread_id=thread_id, namespace=ns)
            existing_texts_lower = {m.text.lower() for m in existing}
            
            for slot, fact in extracted.items():
                # Build a storable sentence from the fact
                fact_text = f"User's {slot.replace('_', ' ')} is {fact.value}"
                
                # Skip if we already have this exact fact
                if fact_text.lower() not in existing_texts_lower:
                    # Check if any existing memory already covers this slot
                    # with the same value (avoid near-duplicates)
                    already_known = False
                    for mem in existing:
                        mem_facts = extract_fact_slots(mem.text)
                        if slot in mem_facts and mem_facts[slot].normalized == fact.normalized:
                            already_known = True
                            break
                    
                    if not already_known:
                        stored = store.store(
                            text=fact_text,
                            thread_id=thread_id,
                            source="inferred",
                            namespace=ns,
                        )
                        auto_learned[slot] = {
                            "value": fact.value,
                            "memory_id": stored.id,
                        }

    # ── Standard memory query ──
    memories = store.query(
        query=query, thread_id=thread_id,
        namespace=ns, include_global=include_global,
    )

    if not memories and not auto_learned:
        result = {
            "found": 0,
            "memories": [],
            "contradictions": [],
            "namespace": ns,
            "auto_learned": auto_learned,
            "note": "No memories stored for this thread yet.",
        }
        return json.dumps(result, indent=2)
    
    # Re-query if we just learned something (so new facts appear in results)
    if auto_learned:
        memories = store.query(
            query=query, thread_id=thread_id,
            namespace=ns, include_global=include_global,
        )

    # Check for internal contradictions among retrieved memories
    combined_text = " ".join(m.text for m in memories)
    report = verifier.verify(combined_text, memories, mode="permissive")

    result = {
        "found": len(memories),
        "namespace": ns,
        "auto_learned": auto_learned,
        "memories": [
            {
                "id": m.id,
                "text": m.text,
                "trust": m.trust,
                "timestamp": m.timestamp,
                "namespace": (m.metadata or {}).get("namespace", ns),
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
    namespace: str = "",
    include_global: bool = True,
) -> str:
    """Verify your draft response against stored memories before sending it.
    
    Returns hallucination list, corrected text, and confidence score.
    Call this before every response that references user facts.
    
    Args:
        draft: The draft response text to verify.
        thread_id: Thread/conversation ID.
        mode: 'strict' rewrites hallucinations; 'permissive' reports only.
        namespace: Project scope. Leave empty for server default.
        include_global: Whether to also verify against 'global' namespace (default True).
    """
    ns = namespace or _default_namespace
    store = _get_store()
    verifier = _get_verifier()

    memories = store.get_all(
        thread_id=thread_id, namespace=ns, include_global=include_global,
    )

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
        "--namespace", "-n",
        default="default",
        help=(
            "Default namespace for this server instance. "
            "Use a project name (e.g. 'my-app') so each project gets "
            "its own memory scope. User-level facts stored with "
            "namespace='global' are visible across all projects. "
            "(default: 'default')"
        ),
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

    # Initialize global store and namespace
    global _store, _default_namespace
    _store = MemoryStore(args.db)
    _default_namespace = args.namespace
    logger.info(
        f"GroundCheck MCP server started with db={args.db}, namespace={args.namespace}"
    )

    # Run via stdio (standard for MCP)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
