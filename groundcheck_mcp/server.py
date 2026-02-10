"""GroundCheck MCP Server — exposes store/check/verify tools via Model Context Protocol.

Usage:
    groundcheck-mcp --db .groundcheck/memory.db
    
    # Or via Python:
    python -m groundcheck_mcp.server --db .groundcheck/memory.db
"""

import argparse
import json
import sys
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import run_stdio
from mcp.types import Tool, TextContent

from groundcheck import GroundCheck, Memory, VerificationReport
from groundcheck.fact_extractor import extract_fact_slots
from .storage import MemoryStore

logger = logging.getLogger("groundcheck-mcp")

# Global state
_store: MemoryStore | None = None
_verifier: GroundCheck | None = None


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


def create_server() -> Server:
    """Create and configure the MCP server with GroundCheck tools."""
    server = Server("groundcheck")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="crt_store_fact",
                description=(
                    "Store a user fact into persistent memory. Returns the stored memory "
                    "and any contradictions detected against existing memories. "
                    "Call this whenever the user states something about themselves, "
                    "their project, preferences, or environment."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The fact to store (e.g. 'User works at Microsoft')",
                        },
                        "source": {
                            "type": "string",
                            "enum": ["user", "document", "code", "inferred"],
                            "description": "Source of the fact. Affects initial trust score.",
                            "default": "user",
                        },
                        "thread_id": {
                            "type": "string",
                            "description": "Thread/conversation ID for memory isolation.",
                            "default": "default",
                        },
                    },
                    "required": ["text"],
                },
            ),
            Tool(
                name="crt_check_memory",
                description=(
                    "Query stored facts before answering. Returns relevant memories "
                    "with trust scores and any detected contradictions. "
                    "Call this before answering questions about the user or project."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for in memory (e.g. 'database technology')",
                        },
                        "thread_id": {
                            "type": "string",
                            "description": "Thread/conversation ID.",
                            "default": "default",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="crt_verify_output",
                description=(
                    "Verify your draft response against stored memories before sending. "
                    "Returns hallucination list, corrected text, and confidence score. "
                    "Call this before every response that references user facts."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "draft": {
                            "type": "string",
                            "description": "The draft response text to verify.",
                        },
                        "thread_id": {
                            "type": "string",
                            "description": "Thread/conversation ID.",
                            "default": "default",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["strict", "permissive"],
                            "description": "strict = rewrite hallucinations; permissive = report only.",
                            "default": "strict",
                        },
                    },
                    "required": ["draft"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        store = _get_store()
        verifier = _get_verifier()

        if name == "crt_store_fact":
            return await _handle_store_fact(store, verifier, arguments)
        elif name == "crt_check_memory":
            return await _handle_check_memory(store, verifier, arguments)
        elif name == "crt_verify_output":
            return await _handle_verify_output(store, verifier, arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    return server


async def _handle_store_fact(
    store: MemoryStore, verifier: GroundCheck, args: dict
) -> list[TextContent]:
    """Store a fact and check for contradictions against existing memories."""
    text = args["text"]
    source = args.get("source", "user")
    thread_id = args.get("thread_id", "default")

    # Store the new memory
    new_mem = store.store(text=text, thread_id=thread_id, source=source)

    # Check for contradictions against all existing memories
    all_memories = store.get_all(thread_id=thread_id)
    
    # Extract facts from the new text
    new_facts = extract_fact_slots(text)
    
    # Run verification against existing memories (excluding the one we just stored)
    existing = [m for m in all_memories if m.id != new_mem.id]
    
    contradictions = []
    if existing and new_facts:
        # Verify the new fact text against existing memories
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

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _handle_check_memory(
    store: MemoryStore, verifier: GroundCheck, args: dict
) -> list[TextContent]:
    """Query memories and return them with contradiction warnings."""
    query = args["query"]
    thread_id = args.get("thread_id", "default")

    memories = store.query(query=query, thread_id=thread_id)

    if not memories:
        result = {
            "found": 0,
            "memories": [],
            "contradictions": [],
            "note": "No memories stored for this thread yet.",
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    # Check for internal contradictions among retrieved memories
    # Use a neutral query to detect cross-memory conflicts
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

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _handle_verify_output(
    store: MemoryStore, verifier: GroundCheck, args: dict
) -> list[TextContent]:
    """Verify a draft response against stored memories."""
    draft = args["draft"]
    thread_id = args.get("thread_id", "default")
    mode = args.get("mode", "strict")

    memories = store.get_all(thread_id=thread_id)

    if not memories:
        result = {
            "passed": True,
            "note": "No memories to verify against. Passing by default.",
            "confidence": 0.0,
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    report = verifier.verify(draft, memories, mode=mode)
    result = _report_to_dict(report)

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


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

    server = create_server()
    import asyncio
    asyncio.run(run_stdio(server))


if __name__ == "__main__":
    main()
