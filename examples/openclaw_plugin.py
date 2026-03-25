#!/usr/bin/env python3
"""
OpenClaw Plugin Skeleton — GroundCheck as a Pipe/Filter Function.

Shows how GroundCheck would plug into Open WebUI as a pipe function
that verifies LLM outputs against stored memories before they reach the user.

This is a skeleton — adapt to your Open WebUI installation.
"""

from typing import Optional


class Pipeline:
    """Open WebUI pipe function for GroundCheck verification."""

    class Valves:
        """Configuration knobs exposed in the Open WebUI admin panel."""
        db_path: str = "~/.groundcheck/memory.db"
        trust_threshold: float = 0.5
        auto_correct: bool = True
        transparency_level: str = "balanced"  # minimal | balanced | audit_heavy

    def __init__(self):
        self.valves = self.Valves()
        self._verifier = None
        self._ledger = None

    def _ensure_loaded(self):
        if self._verifier is not None:
            return
        from groundcheck import GroundCheck, ContradictionLedger
        self._verifier = GroundCheck()
        self._ledger = ContradictionLedger(db_path=self.valves.db_path)

    def pipe(self, body: dict) -> dict:
        """Main pipe function — called for every LLM response.

        Args:
            body: Open WebUI message body with 'messages' list.

        Returns:
            Modified body with verified/corrected assistant message.
        """
        self._ensure_loaded()

        messages = body.get("messages", [])
        if not messages:
            return body

        # Get the latest assistant message
        last = messages[-1]
        if last.get("role") != "assistant":
            return body

        generated_text = last.get("content", "")
        if not generated_text:
            return body

        # Load memories from ledger DB
        # (In production, query from your memory store)
        from groundcheck import Memory
        memories = self._load_memories()

        if not memories:
            return body

        # Verify
        mode = "strict" if self.valves.auto_correct else "permissive"
        result = self._verifier.verify(generated_text, memories, mode=mode)

        if not result.passed:
            # Attach metadata
            if result.corrected and self.valves.auto_correct:
                last["content"] = result.corrected

            # Add disclosure note if needed
            if result.requires_disclosure and result.contradiction_details:
                slots = [cd.slot for cd in result.contradiction_details]
                note = f"\n\n> GroundCheck: conflicting facts detected in {', '.join(slots)}."
                last["content"] = last.get("content", "") + note

        return body

    def _load_memories(self):
        """Load memories from SQLite. Adapt to your schema."""
        from groundcheck import Memory
        import sqlite3

        try:
            conn = sqlite3.connect(self.valves.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, text, trust FROM memories WHERE trust > ? ORDER BY trust DESC LIMIT 100",
                (self.valves.trust_threshold,),
            ).fetchall()
            conn.close()
            return [Memory(id=r["id"], text=r["text"], trust=r["trust"]) for r in rows]
        except Exception:
            return []
