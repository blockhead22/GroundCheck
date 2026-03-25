"""
CRT Contradiction Ledger — No Silent Overwrites.

Append-only SQLite ledger tracking every contradiction detected by the system.
Nothing is deleted or silently replaced. Tension is preserved until reflection.

Philosophy:
- Contradictions are signals, not bugs
- Nothing is deleted or silently replaced
- Tension is preserved until reflection
- History matters more than consistency
"""

import sqlite3
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
        db_path: SQLite database path. Defaults to ``~/.groundcheck/ledger.db``
            or ``GROUNDCHECK_DB`` env var.
        config: CRT configuration for math operations.
        fact_extractor: Optional callable ``(text) -> dict`` for extracting
            fact slots from text. If not provided, uses groundcheck's built-in
            ``extract_fact_slots``.
        drift_assessor: Optional callable ``(old_text, new_text) -> str``
            returning a contradiction type string. Replaces the monolith's
            LLM drift assessor.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        config: Optional[CRTConfig] = None,
        fact_extractor=None,
        drift_assessor=None,
    ):
        self.db_path = db_path or _default_db_path()
        self.config = config or CRTConfig()
        self.crt_math = CRTMath(self.config)
        self.default_thread_id: Optional[str] = None

        # Pluggable fact extraction
        self._fact_extractor = fact_extractor
        # Pluggable drift assessor (replaces LLM drift assessor)
        self._drift_assessor = drift_assessor

        self._init_db()

    def _extract_all_facts(self, text: str) -> Dict[str, Any]:
        """Extract facts from text using configured extractor."""
        if self._fact_extractor is not None:
            try:
                return self._fact_extractor(text) or {}
            except Exception:
                pass

        # Fall back to groundcheck's built-in extractor
        try:
            from .fact_extractor import extract_fact_slots
            return extract_fact_slots(text) or {}
        except ImportError:
            return {}

    def _get_connection(self, timeout: float = 30.0) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=timeout, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _has_contradiction_thread_column(self) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("PRAGMA table_info(contradictions)")
            return any(str(row[1] or "") == "thread_id" for row in cursor.fetchall())
        finally:
            conn.close()

    def _init_db(self):
        # Ensure parent directory exists
        from pathlib import Path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contradictions (
                ledger_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                old_memory_id TEXT NOT NULL,
                new_memory_id TEXT NOT NULL,
                drift_mean REAL NOT NULL,
                drift_reason REAL,
                confidence_delta REAL,
                status TEXT NOT NULL,
                contradiction_type TEXT DEFAULT 'conflict',
                affects_slots TEXT,
                query TEXT,
                summary TEXT,
                resolution_timestamp REAL,
                resolution_method TEXT,
                merged_memory_id TEXT,
                metadata TEXT,
                thread_id TEXT
            )
        """)

        # Migration columns
        for col, coltype in [
            ("metadata", "TEXT"),
            ("thread_id", "TEXT"),
            ("lifecycle_state", "TEXT DEFAULT 'active'"),
            ("confirmation_count", "INTEGER DEFAULT 0"),
            ("disclosure_count", "INTEGER DEFAULT 0"),
            ("settled_at", "REAL"),
            ("archived_at", "REAL"),
        ]:
            try:
                cursor.execute(f"ALTER TABLE contradictions ADD COLUMN {col} {coltype}")
            except sqlite3.OperationalError:
                pass

        # Indexes
        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_contradictions_status ON contradictions(status)",
            "CREATE INDEX IF NOT EXISTS idx_contradictions_old_memory ON contradictions(old_memory_id)",
            "CREATE INDEX IF NOT EXISTS idx_contradictions_new_memory ON contradictions(new_memory_id)",
            "CREATE INDEX IF NOT EXISTS idx_contradictions_thread_status ON contradictions(thread_id, status, timestamp)",
        ]:
            try:
                cursor.execute(idx_sql)
            except Exception:
                pass

        # Reflection queue
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reflection_queue (
                queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                ledger_id TEXT NOT NULL,
                volatility REAL NOT NULL,
                priority TEXT NOT NULL,
                context_json TEXT,
                processed INTEGER DEFAULT 0,
                FOREIGN KEY (ledger_id) REFERENCES contradictions(ledger_id)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_reflection_queue_processed
            ON reflection_queue(processed, priority)
        """)

        # Worklog
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contradiction_worklog (
                ledger_id TEXT PRIMARY KEY,
                first_asked_at REAL,
                last_asked_at REAL,
                ask_count INTEGER DEFAULT 0,
                last_user_answer TEXT,
                last_user_answer_at REAL
            )
        """)

        # Conflict resolutions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conflict_resolutions (
                ledger_id TEXT PRIMARY KEY,
                resolution_method TEXT NOT NULL,
                chosen_memory_id TEXT,
                user_feedback TEXT,
                timestamp REAL NOT NULL,
                FOREIGN KEY (ledger_id) REFERENCES contradictions(ledger_id)
            )
        """)

        conn.commit()
        conn.close()

    # ========================================================================
    # Recording
    # ========================================================================

    def _classify_contradiction(
        self, old_text: str, new_text: str, drift_mean: float,
        old_vector=None, new_vector=None,
    ) -> str:
        """Classify contradiction type."""
        # Try pluggable drift assessor first
        if self._drift_assessor is not None:
            try:
                result = self._drift_assessor(old_text, new_text)
                if result:
                    return result
            except Exception:
                pass

        # Use CRTMath's classify_fact_change
        old_facts = self._extract_all_facts(old_text) or {}
        new_facts = self._extract_all_facts(new_text) or {}
        shared_slots = set(old_facts.keys()) & set(new_facts.keys())

        if shared_slots:
            slot = sorted(shared_slots)[0]
            old_val = str(getattr(old_facts.get(slot), "value", old_facts.get(slot, "")))
            new_val = str(getattr(new_facts.get(slot), "value", new_facts.get(slot, "")))
            return self.crt_math.classify_fact_change(slot, new_val, old_val, new_text, old_text)

        # Heuristic classification
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

        # Extract affected slots
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

        conn = self._get_connection()
        cursor = conn.cursor()

        metadata = {}
        if suggested_policy:
            metadata["suggested_policy"] = suggested_policy

        cursor.execute("""
            INSERT INTO contradictions
            (ledger_id, timestamp, old_memory_id, new_memory_id, drift_mean,
             drift_reason, confidence_delta, status, contradiction_type,
             affects_slots, query, summary, metadata, thread_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.ledger_id, entry.timestamp,
            old_memory_id, new_memory_id,
            drift_mean, drift_reason, confidence_delta,
            entry.status, entry.contradiction_type,
            entry.affects_slots, query, entry.summary,
            json.dumps(metadata) if metadata else None,
            entry.thread_id,
        ))

        conn.commit()
        conn.close()
        return entry

    # ========================================================================
    # Queries
    # ========================================================================

    def get_open_contradictions(
        self, limit: int = 10, thread_id: Optional[str] = None,
    ) -> List[ContradictionEntry]:
        conn = self._get_connection()
        cursor = conn.cursor()
        effective_thread = thread_id if thread_id is not None else self.default_thread_id
        if effective_thread is not None and self._has_contradiction_thread_column():
            cursor.execute("""
                SELECT * FROM contradictions
                WHERE status = ?
                  AND COALESCE(thread_id, 'default') = ?
                ORDER BY timestamp DESC LIMIT ?
            """, (ContradictionStatus.OPEN, str(effective_thread), limit))
        else:
            cursor.execute("""
                SELECT * FROM contradictions
                WHERE status = ?
                ORDER BY timestamp DESC LIMIT ?
            """, (ContradictionStatus.OPEN, limit))
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_entry(row) for row in rows]

    def get_all_contradictions(
        self, limit: int = 100, thread_id: Optional[str] = None,
    ) -> List[ContradictionEntry]:
        conn = self._get_connection()
        cursor = conn.cursor()
        effective_thread = thread_id if thread_id is not None else self.default_thread_id
        if effective_thread is not None and self._has_contradiction_thread_column():
            cursor.execute("""
                SELECT * FROM contradictions
                WHERE COALESCE(thread_id, 'default') = ?
                ORDER BY timestamp DESC LIMIT ?
            """, (str(effective_thread), int(limit)))
        else:
            cursor.execute("""
                SELECT * FROM contradictions
                ORDER BY timestamp DESC LIMIT ?
            """, (int(limit),))
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_entry(row) for row in rows]

    def get_contradiction_by_memory(self, memory_id: str) -> List[ContradictionEntry]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM contradictions
            WHERE old_memory_id = ? OR new_memory_id = ?
            ORDER BY timestamp DESC
        """, (memory_id, memory_id))
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_entry(row) for row in rows]

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
        """Mark contradiction as resolved."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE contradictions
            SET status = ?, resolution_timestamp = ?,
                resolution_method = ?, merged_memory_id = ?
            WHERE ledger_id = ?
        """, (new_status, time.time(), method, merged_memory_id, ledger_id))
        conn.commit()
        conn.close()

    # ========================================================================
    # Worklog (ask/answer tracking)
    # ========================================================================

    def mark_contradiction_asked(self, ledger_id: str) -> None:
        ts = time.time()
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO contradiction_worklog (ledger_id, first_asked_at, last_asked_at, ask_count)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(ledger_id) DO UPDATE SET
                last_asked_at=excluded.last_asked_at,
                ask_count=coalesce(contradiction_worklog.ask_count, 0) + 1
        """, (ledger_id, ts, ts))
        conn.commit()
        conn.close()

    def record_contradiction_user_answer(self, ledger_id: str, answer: str) -> None:
        ts = time.time()
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO contradiction_worklog (ledger_id, last_user_answer, last_user_answer_at)
            VALUES (?, ?, ?)
            ON CONFLICT(ledger_id) DO UPDATE SET
                last_user_answer=excluded.last_user_answer,
                last_user_answer_at=excluded.last_user_answer_at
        """, (ledger_id, (answer or "").strip(), ts))
        conn.commit()
        conn.close()

    # ========================================================================
    # Reflection Queue
    # ========================================================================

    def queue_reflection(
        self, ledger_id: str, volatility: float, context: Optional[Dict] = None,
    ):
        """Queue contradiction for reflection."""
        if volatility >= 0.7:
            priority = "high"
        elif volatility >= 0.4:
            priority = "medium"
        else:
            priority = "low"

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO reflection_queue
            (timestamp, ledger_id, volatility, priority, context_json)
            VALUES (?, ?, ?, ?, ?)
        """, (time.time(), ledger_id, volatility, priority,
              json.dumps(context) if context else None))
        conn.commit()
        conn.close()

    def get_reflection_queue(self, priority: Optional[str] = None) -> List[Dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        if priority:
            cursor.execute("""
                SELECT * FROM reflection_queue
                WHERE processed = 0 AND priority = ?
                ORDER BY volatility DESC, timestamp ASC
            """, (priority,))
        else:
            cursor.execute("""
                SELECT * FROM reflection_queue
                WHERE processed = 0
                ORDER BY
                    CASE priority
                        WHEN 'high' THEN 1
                        WHEN 'medium' THEN 2
                        WHEN 'low' THEN 3
                    END,
                    volatility DESC, timestamp ASC
            """)
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "queue_id": row[0],
                "timestamp": row[1],
                "ledger_id": row[2],
                "volatility": row[3],
                "priority": row[4],
                "context": json.loads(row[5]) if row[5] else None,
            }
            for row in rows
        ]

    def mark_reflection_processed(self, queue_id: int):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE reflection_queue SET processed = 1 WHERE queue_id = ?",
            (queue_id,),
        )
        conn.commit()
        conn.close()

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_contradiction_stats(
        self, days: int = 7, thread_id: Optional[str] = None,
    ) -> Dict:
        since = time.time() - (days * 86400)
        effective_thread = thread_id if thread_id is not None else self.default_thread_id

        conn = self._get_connection()
        cursor = conn.cursor()
        has_thread = effective_thread is not None and self._has_contradiction_thread_column()
        tc = " AND COALESCE(thread_id, 'default') = ?" if has_thread else ""
        tp = (str(effective_thread),) if has_thread else ()

        cursor.execute(
            f"SELECT COUNT(*) FROM contradictions WHERE timestamp > ?{tc}",
            (since,) + tp,
        )
        total = cursor.fetchone()[0]

        cursor.execute(f"""
            SELECT status, COUNT(*)
            FROM contradictions
            WHERE timestamp > ?{tc}
            GROUP BY status
        """, (since,) + tp)
        by_status = dict(cursor.fetchall())

        cursor.execute(
            f"SELECT AVG(drift_mean) FROM contradictions WHERE timestamp > ?{tc}",
            (since,) + tp,
        )
        avg_drift = cursor.fetchone()[0] or 0.0

        cursor.execute("SELECT COUNT(*) FROM reflection_queue WHERE processed = 0")
        pending = cursor.fetchone()[0]

        conn.close()

        return {
            "total_contradictions": total,
            "open": by_status.get(ContradictionStatus.OPEN, 0),
            "resolved": by_status.get(ContradictionStatus.RESOLVED, 0),
            "accepted": by_status.get(ContradictionStatus.ACCEPTED, 0),
            "average_drift": avg_drift,
            "pending_reflections": pending,
            "days": days,
        }

    # ========================================================================
    # Lifecycle State Management
    # ========================================================================

    def process_lifecycle_transitions(self) -> int:
        """Process lifecycle state transitions for all active contradictions.

        Returns the number of transitions performed.
        """
        from .lifecycle import ContradictionLifecycle, ContradictionLifecycleEntry, ContradictionLifecycleState

        lifecycle = ContradictionLifecycle()
        transitions = 0

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT ledger_id, lifecycle_state, timestamp, confirmation_count, disclosure_count
            FROM contradictions
            WHERE status = ? AND lifecycle_state != 'archived'
        """, (ContradictionStatus.OPEN,))

        rows = cursor.fetchall()
        conn.close()

        for row in rows:
            lid, current_state, detected_at, conf_count, disc_count = row
            entry = ContradictionLifecycleEntry(
                ledger_id=lid,
                state=ContradictionLifecycleState(current_state or "active"),
                detected_at=detected_at or time.time(),
                confirmation_count=conf_count or 0,
                disclosure_count=disc_count or 0,
            )
            new_state = lifecycle.update_state(entry)
            if new_state != entry.state:
                self._update_lifecycle_in_db(lid, new_state.value)
                transitions += 1

        return transitions

    def _update_lifecycle_in_db(self, ledger_id: str, new_state: str):
        conn = self._get_connection()
        cursor = conn.cursor()
        updates = ["lifecycle_state = ?"]
        params: list = [new_state]
        if new_state == "settled":
            updates.append("settled_at = ?")
            params.append(time.time())
        elif new_state == "archived":
            updates.append("archived_at = ?")
            params.append(time.time())
        params.append(ledger_id)
        cursor.execute(
            f"UPDATE contradictions SET {', '.join(updates)} WHERE ledger_id = ?",
            params,
        )
        conn.commit()
        conn.close()

    # ========================================================================
    # Helpers
    # ========================================================================

    def _row_to_entry(self, row) -> ContradictionEntry:
        return ContradictionEntry(
            ledger_id=row[0],
            timestamp=row[1],
            old_memory_id=row[2],
            new_memory_id=row[3],
            drift_mean=row[4],
            drift_reason=row[5],
            confidence_delta=row[6],
            status=row[7],
            contradiction_type=row[8] if len(row) > 8 else ContradictionType.CONFLICT,
            affects_slots=row[9] if len(row) > 9 else None,
            query=row[10] if len(row) > 10 else None,
            summary=row[11] if len(row) > 11 else None,
            resolution_timestamp=row[12] if len(row) > 12 else None,
            resolution_method=row[13] if len(row) > 13 else None,
            merged_memory_id=row[14] if len(row) > 14 else None,
            thread_id=row[16] if len(row) > 16 else None,
        )
