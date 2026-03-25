"""
CRT Contradiction Ledger — No Silent Overwrites.

Append-only ledger tracking every contradiction detected by the system.
Nothing is deleted or silently replaced. Tension is preserved until reflection.

Supports pluggable storage backends via the ``LedgerBackend`` protocol.
Default: SQLite. For testing: ``InMemoryBackend``.

Usage::

    from groundcheck.ledger import ContradictionLedger
    from groundcheck.backends import InMemoryBackend

    # Production (SQLite, default)
    ledger = ContradictionLedger()

    # Testing (in-memory, zero cleanup)
    ledger = ContradictionLedger(backend=InMemoryBackend())
"""

import json
import logging
import time
import os
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from .trust_math import CRTMath, CRTConfig, MemorySource

logger = logging.getLogger(__name__)


def _default_db_path() -> str:
    """Return default ledger DB path, respecting GROUNDCHECK_DB env var."""
    env_path = os.environ.get("GROUNDCHECK_DB")
    if env_path:
        return env_path
    from pathlib import Path
    return str(Path.home() / ".groundcheck" / "ledger.db")


class ContradictionStatus:
    """Status of contradiction resolution."""
    OPEN = "open"
    REFLECTING = "reflecting"
    RESOLVED = "resolved"
    ACCEPTED = "accepted"


class ContradictionType:
    """Type of contradiction based on fact topology."""
    REFINEMENT = "refinement"
    REVISION = "revision"
    TEMPORAL = "temporal"
    CONFLICT = "conflict"
    DENIAL = "denial"


@dataclass
class ContradictionEntry:
    """A contradiction ledger entry."""
    ledger_id: str
    timestamp: float
    old_memory_id: str
    new_memory_id: str

    # Drift measurements
    drift_mean: float
    drift_reason: Optional[float] = None
    confidence_delta: float = 0.0

    # Status
    status: str = ContradictionStatus.OPEN
    contradiction_type: str = ContradictionType.CONFLICT

    # Slot tracking
    affects_slots: Optional[str] = None

    # Metadata
    query: Optional[str] = None
    summary: Optional[str] = None
    resolution_timestamp: Optional[float] = None
    resolution_method: Optional[str] = None
    merged_memory_id: Optional[str] = None
    thread_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "ledger_id": self.ledger_id,
            "timestamp": self.timestamp,
            "old_memory_id": self.old_memory_id,
            "new_memory_id": self.new_memory_id,
            "drift_mean": self.drift_mean,
            "drift_reason": self.drift_reason,
            "confidence_delta": self.confidence_delta,
            "status": self.status,
            "contradiction_type": self.contradiction_type,
            "affects_slots": self.affects_slots,
            "query": self.query,
            "summary": self.summary,
            "resolution_timestamp": self.resolution_timestamp,
            "resolution_method": self.resolution_method,
            "merged_memory_id": self.merged_memory_id,
            "thread_id": self.thread_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContradictionEntry":
        return cls(
            ledger_id=data["ledger_id"],
            timestamp=data.get("timestamp", time.time()),
            old_memory_id=data.get("old_memory_id", ""),
            new_memory_id=data.get("new_memory_id", ""),
            drift_mean=data.get("drift_mean", 0.0),
            drift_reason=data.get("drift_reason"),
            confidence_delta=data.get("confidence_delta", 0.0),
            status=data.get("status", ContradictionStatus.OPEN),
            contradiction_type=data.get("contradiction_type", ContradictionType.CONFLICT),
            affects_slots=data.get("affects_slots"),
            query=data.get("query"),
            summary=data.get("summary"),
            resolution_timestamp=data.get("resolution_timestamp"),
            resolution_method=data.get("resolution_method"),
            merged_memory_id=data.get("merged_memory_id"),
            thread_id=data.get("thread_id"),
        )


class ContradictionLedger:
    """CRT contradiction ledger system.

    NO SILENT OVERWRITES. When beliefs diverge:
    1. Create ledger entry
    2. Preserve both old and new
    3. Log drift measurements
    4. Track resolution status
    5. Trigger reflection if needed

    Args:
        db_path: SQLite database path. Ignored when ``backend`` is provided.
        config: CRT configuration for math operations.
        fact_extractor: Optional callable ``(text) -> dict``.
        drift_assessor: Optional callable ``(old_text, new_text) -> str``.
        backend: Storage backend (``LedgerBackend`` protocol). Defaults to
            ``SQLiteBackend(db_path)``. Pass ``InMemoryBackend()`` for tests.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        config: Optional[CRTConfig] = None,
        fact_extractor=None,
        drift_assessor=None,
        backend=None,
    ):
        self.config = config or CRTConfig()
        self.crt_math = CRTMath(self.config)
        self.default_thread_id: Optional[str] = None

        # Pluggable fact extraction
        self._fact_extractor = fact_extractor
        # Pluggable drift assessor
        self._drift_assessor = drift_assessor

        # Storage backend
        if backend is not None:
            self._backend = backend
        else:
            from .backends import SQLiteBackend
            self._backend = SQLiteBackend(db_path=db_path or _default_db_path())

        # Backward compat: expose db_path if using SQLite
        self.db_path = getattr(self._backend, "db_path", None)

        self._backend.init_storage()

    @property
    def backend(self):
        """The active storage backend."""
        return self._backend

    def _extract_all_facts(self, text: str) -> Dict[str, Any]:
        """Extract facts from text using configured extractor."""
        if self._fact_extractor is not None:
            try:
                return self._fact_extractor(text) or {}
            except Exception:
                pass
        try:
            from .fact_extractor import extract_fact_slots
            return extract_fact_slots(text) or {}
        except ImportError:
            return {}

    # ========================================================================
    # Recording
    # ========================================================================

    def _classify_contradiction(
        self, old_text: str, new_text: str, drift_mean: float,
        old_vector=None, new_vector=None,
    ) -> str:
        if self._drift_assessor is not None:
            try:
                result = self._drift_assessor(old_text, new_text)
                if result:
                    return result
            except Exception:
                pass

        old_facts = self._extract_all_facts(old_text) or {}
        new_facts = self._extract_all_facts(new_text) or {}
        shared_slots = set(old_facts.keys()) & set(new_facts.keys())

        if shared_slots:
            slot = sorted(shared_slots)[0]
            old_val = str(getattr(old_facts.get(slot), "value", old_facts.get(slot, "")))
            new_val = str(getattr(new_facts.get(slot), "value", new_facts.get(slot, "")))
            return self.crt_math.classify_fact_change(slot, new_val, old_val, new_text, old_text)

        old_lower = old_text.lower()
        new_lower = new_text.lower()

        revision_keywords = ["actually", "correction", "i meant", "not ", "wrong", "mistake"]
        if any(kw in new_lower for kw in revision_keywords):
            return ContradictionType.REVISION

        if old_text in new_text or new_text in old_text:
            return ContradictionType.REFINEMENT

        temporal_markers = [
            "now", "currently", "recently", "switched", "changed",
            "moved", "started", "used to", "no longer", "promoted",
        ]
        if any(m in new_lower or m in old_lower for m in temporal_markers):
            return ContradictionType.TEMPORAL

        if old_vector is not None and new_vector is not None:
            similarity = self.crt_math.similarity(old_vector, new_vector)
            if 0.7 <= similarity < 0.9:
                return ContradictionType.REFINEMENT

        return ContradictionType.CONFLICT

    def _generate_summary(
        self, drift: float, conf_delta: float,
        contradiction_type: str = ContradictionType.CONFLICT,
    ) -> str:
        if drift > 0.5:
            intensity = "Strong"
        elif drift > 0.3:
            intensity = "Moderate"
        else:
            intensity = "Mild"

        type_desc = {
            ContradictionType.REFINEMENT: "Refinement",
            ContradictionType.REVISION: "Revision",
            ContradictionType.TEMPORAL: "Temporal progression",
            ContradictionType.CONFLICT: "Conflict",
        }.get(contradiction_type, "Contradiction")

        conf_desc = ""
        if conf_delta > 0.3:
            conf_desc = "with significant confidence shift"
        elif conf_delta > 0.1:
            conf_desc = "with moderate confidence shift"

        return f"{type_desc}: {intensity} belief divergence (drift={drift:.2f}) {conf_desc}".strip()

    def record_contradiction(
        self,
        old_memory_id: str,
        new_memory_id: str,
        drift_mean: float,
        confidence_delta: float,
        query: Optional[str] = None,
        summary: Optional[str] = None,
        drift_reason: Optional[float] = None,
        old_text: Optional[str] = None,
        new_text: Optional[str] = None,
        old_vector=None,
        new_vector=None,
        contradiction_type: Optional[str] = None,
        suggested_policy: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> ContradictionEntry:
        """Record a contradiction event. NO DELETION. NO REPLACEMENT."""
        if contradiction_type is None:
            if old_text and new_text:
                contradiction_type = self._classify_contradiction(
                    old_text, new_text, drift_mean, old_vector, new_vector,
                )
            else:
                contradiction_type = ContradictionType.CONFLICT

        affects_slots_set: set = set()
        if old_text and new_text:
            old_facts = self._extract_all_facts(old_text) or {}
            new_facts = self._extract_all_facts(new_text) or {}
            affects_slots_set = set(old_facts.keys()) & set(new_facts.keys())

        affects_slots_str = ",".join(sorted(affects_slots_set)) if affects_slots_set else None

        entry = ContradictionEntry(
            ledger_id=f"contra_{int(time.time() * 1000)}_{hash(old_memory_id + new_memory_id) % 10000}",
            timestamp=time.time(),
            old_memory_id=old_memory_id,
            new_memory_id=new_memory_id,
            drift_mean=drift_mean,
            drift_reason=drift_reason,
            confidence_delta=confidence_delta,
            status=ContradictionStatus.OPEN,
            contradiction_type=contradiction_type,
            affects_slots=affects_slots_str,
            query=query,
            summary=summary or self._generate_summary(drift_mean, confidence_delta, contradiction_type),
            thread_id=thread_id or self.default_thread_id,
        )

        metadata = {}
        if suggested_policy:
            metadata["suggested_policy"] = suggested_policy

        self._backend.insert_contradiction({
            "ledger_id": entry.ledger_id,
            "timestamp": entry.timestamp,
            "old_memory_id": old_memory_id,
            "new_memory_id": new_memory_id,
            "drift_mean": drift_mean,
            "drift_reason": drift_reason,
            "confidence_delta": confidence_delta,
            "status": entry.status,
            "contradiction_type": entry.contradiction_type,
            "affects_slots": entry.affects_slots,
            "query": query,
            "summary": entry.summary,
            "metadata": metadata if metadata else None,
            "thread_id": entry.thread_id,
        })

        return entry

    # ========================================================================
    # Queries
    # ========================================================================

    def _dict_to_entry(self, d: Dict[str, Any]) -> ContradictionEntry:
        return ContradictionEntry(
            ledger_id=d.get("ledger_id", ""),
            timestamp=d.get("timestamp", 0.0),
            old_memory_id=d.get("old_memory_id", ""),
            new_memory_id=d.get("new_memory_id", ""),
            drift_mean=d.get("drift_mean", 0.0),
            drift_reason=d.get("drift_reason"),
            confidence_delta=d.get("confidence_delta", 0.0),
            status=d.get("status", ContradictionStatus.OPEN),
            contradiction_type=d.get("contradiction_type", ContradictionType.CONFLICT),
            affects_slots=d.get("affects_slots"),
            query=d.get("query"),
            summary=d.get("summary"),
            resolution_timestamp=d.get("resolution_timestamp"),
            resolution_method=d.get("resolution_method"),
            merged_memory_id=d.get("merged_memory_id"),
            thread_id=d.get("thread_id"),
        )

    def get_open_contradictions(
        self, limit: int = 10, thread_id: Optional[str] = None,
    ) -> List[ContradictionEntry]:
        effective_thread = thread_id if thread_id is not None else self.default_thread_id
        rows = self._backend.get_contradictions(
            status=ContradictionStatus.OPEN,
            thread_id=effective_thread,
            limit=limit,
        )
        return [self._dict_to_entry(d) for d in rows]

    def get_all_contradictions(
        self, limit: int = 100, thread_id: Optional[str] = None,
    ) -> List[ContradictionEntry]:
        effective_thread = thread_id if thread_id is not None else self.default_thread_id
        rows = self._backend.get_contradictions(
            thread_id=effective_thread,
            limit=limit,
        )
        return [self._dict_to_entry(d) for d in rows]

    def get_contradiction_by_memory(self, memory_id: str) -> List[ContradictionEntry]:
        rows = self._backend.get_contradictions(memory_id=memory_id)
        return [self._dict_to_entry(d) for d in rows]

    def has_open_contradiction(self, memory_id: str) -> bool:
        contradictions = self.get_contradiction_by_memory(memory_id)
        return any(c.status == ContradictionStatus.OPEN for c in contradictions)

    # ========================================================================
    # Resolution
    # ========================================================================

    def resolve_contradiction(
        self,
        ledger_id: str,
        method: str,
        merged_memory_id: Optional[str] = None,
        new_status: str = ContradictionStatus.RESOLVED,
    ):
        self._backend.update_contradiction(ledger_id, {
            "status": new_status,
            "resolution_timestamp": time.time(),
            "resolution_method": method,
            "merged_memory_id": merged_memory_id,
        })

    # ========================================================================
    # Worklog
    # ========================================================================

    def mark_contradiction_asked(self, ledger_id: str) -> None:
        self._backend.upsert_worklog_asked(ledger_id, time.time())

    def record_contradiction_user_answer(self, ledger_id: str, answer: str) -> None:
        self._backend.upsert_worklog_answer(ledger_id, answer, time.time())

    # ========================================================================
    # Reflection Queue
    # ========================================================================

    def queue_reflection(
        self, ledger_id: str, volatility: float, context: Optional[Dict] = None,
    ):
        if volatility >= 0.7:
            priority = "high"
        elif volatility >= 0.4:
            priority = "medium"
        else:
            priority = "low"

        self._backend.insert_reflection({
            "timestamp": time.time(),
            "ledger_id": ledger_id,
            "volatility": volatility,
            "priority": priority,
            "context": context,
        })

    def get_reflection_queue(self, priority: Optional[str] = None) -> List[Dict]:
        return self._backend.get_reflections(processed=False, priority=priority)

    def mark_reflection_processed(self, queue_id: int):
        self._backend.mark_reflection_processed(queue_id)

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_contradiction_stats(
        self, days: int = 7, thread_id: Optional[str] = None,
    ) -> Dict:
        since = time.time() - (days * 86400)
        effective_thread = thread_id if thread_id is not None else self.default_thread_id

        total = self._backend.count_contradictions(since=since, thread_id=effective_thread)
        open_count = self._backend.count_contradictions(
            since=since, status=ContradictionStatus.OPEN, thread_id=effective_thread,
        )
        resolved_count = self._backend.count_contradictions(
            since=since, status=ContradictionStatus.RESOLVED, thread_id=effective_thread,
        )
        accepted_count = self._backend.count_contradictions(
            since=since, status=ContradictionStatus.ACCEPTED, thread_id=effective_thread,
        )
        avg = self._backend.avg_drift(since=since, thread_id=effective_thread)
        pending = self._backend.count_pending_reflections()

        return {
            "total_contradictions": total,
            "open": open_count,
            "resolved": resolved_count,
            "accepted": accepted_count,
            "average_drift": avg,
            "pending_reflections": pending,
            "days": days,
        }

    # ========================================================================
    # Lifecycle State Management
    # ========================================================================

    def process_lifecycle_transitions(self) -> int:
        from .lifecycle import ContradictionLifecycle, ContradictionLifecycleEntry, ContradictionLifecycleState

        lifecycle = ContradictionLifecycle()
        transitions = 0

        rows = self._backend.get_contradictions(status=ContradictionStatus.OPEN, limit=1000)

        for d in rows:
            current_state = d.get("lifecycle_state") or "active"
            if current_state == "archived":
                continue

            entry = ContradictionLifecycleEntry(
                ledger_id=d["ledger_id"],
                state=ContradictionLifecycleState(current_state),
                detected_at=d.get("timestamp") or time.time(),
                confirmation_count=d.get("confirmation_count") or 0,
                disclosure_count=d.get("disclosure_count") or 0,
            )
            new_state = lifecycle.update_state(entry)
            if new_state != entry.state:
                updates: Dict[str, Any] = {"lifecycle_state": new_state.value}
                if new_state == ContradictionLifecycleState.SETTLED:
                    updates["settled_at"] = time.time()
                elif new_state == ContradictionLifecycleState.ARCHIVED:
                    updates["archived_at"] = time.time()
                self._backend.update_contradiction(d["ledger_id"], updates)
                transitions += 1

        return transitions
