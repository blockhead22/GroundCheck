"""
Storage backends for the Contradiction Ledger.

Provides:
- ``LedgerBackend`` — abstract protocol that any storage backend must satisfy
- ``SQLiteBackend`` — production default (WAL mode, indexes, migrations)
- ``InMemoryBackend`` — zero-dependency in-memory store for testing and ephemeral use

Usage::

    from groundcheck.backends import InMemoryBackend
    from groundcheck.ledger import ContradictionLedger

    ledger = ContradictionLedger(backend=InMemoryBackend())
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ============================================================================
# Protocol
# ============================================================================


@runtime_checkable
class LedgerBackend(Protocol):
    """Abstract storage interface for the contradiction ledger.

    Implement this protocol to plug in Postgres, Redis, DynamoDB, or any
    other storage backend.  The built-in ``SQLiteBackend`` and
    ``InMemoryBackend`` are reference implementations.
    """

    def init_storage(self) -> None:
        """Create tables / collections / indexes. Idempotent."""
        ...

    def insert_contradiction(self, data: Dict[str, Any]) -> None:
        """Insert a new contradiction row."""
        ...

    def get_contradictions(
        self,
        *,
        status: Optional[str] = None,
        thread_id: Optional[str] = None,
        memory_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query contradictions with optional filters.

        Returns newest-first list of dicts matching the ``ContradictionEntry`` schema.
        """
        ...

    def update_contradiction(self, ledger_id: str, updates: Dict[str, Any]) -> bool:
        """Update fields on a single contradiction row. Returns success."""
        ...

    # Reflection queue
    def insert_reflection(self, data: Dict[str, Any]) -> None:
        ...

    def get_reflections(self, *, processed: bool = False, priority: Optional[str] = None) -> List[Dict[str, Any]]:
        ...

    def mark_reflection_processed(self, queue_id: Any) -> None:
        ...

    # Worklog
    def upsert_worklog_asked(self, ledger_id: str, ts: float) -> None:
        ...

    def upsert_worklog_answer(self, ledger_id: str, answer: str, ts: float) -> None:
        ...

    # Stats helpers
    def count_contradictions(self, *, since: float, status: Optional[str] = None, thread_id: Optional[str] = None) -> int:
        ...

    def avg_drift(self, *, since: float, thread_id: Optional[str] = None) -> float:
        ...

    def count_pending_reflections(self) -> int:
        ...


# ============================================================================
# SQLite Backend
# ============================================================================


class SQLiteBackend:
    """Production SQLite backend with WAL mode and migrations."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            env_path = os.environ.get("GROUNDCHECK_DB")
            if env_path:
                db_path = env_path
            else:
                db_path = str(Path.home() / ".groundcheck" / "ledger.db")
        self.db_path = db_path

    def _get_connection(self, timeout: float = 30.0) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=timeout, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def init_storage(self) -> None:
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
                thread_id TEXT,
                lifecycle_state TEXT DEFAULT 'active',
                confirmation_count INTEGER DEFAULT 0,
                disclosure_count INTEGER DEFAULT 0,
                settled_at REAL,
                archived_at REAL
            )
        """)

        # Migrations for older DBs
        for col, coltype in [
            ("metadata", "TEXT"), ("thread_id", "TEXT"),
            ("lifecycle_state", "TEXT DEFAULT 'active'"),
            ("confirmation_count", "INTEGER DEFAULT 0"),
            ("disclosure_count", "INTEGER DEFAULT 0"),
            ("settled_at", "REAL"), ("archived_at", "REAL"),
        ]:
            try:
                cursor.execute(f"ALTER TABLE contradictions ADD COLUMN {col} {coltype}")
            except sqlite3.OperationalError:
                pass

        for sql in [
            "CREATE INDEX IF NOT EXISTS idx_contradictions_status ON contradictions(status)",
            "CREATE INDEX IF NOT EXISTS idx_contradictions_old_memory ON contradictions(old_memory_id)",
            "CREATE INDEX IF NOT EXISTS idx_contradictions_new_memory ON contradictions(new_memory_id)",
            "CREATE INDEX IF NOT EXISTS idx_contradictions_thread_status ON contradictions(thread_id, status, timestamp)",
        ]:
            try:
                cursor.execute(sql)
            except Exception:
                pass

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

    def insert_contradiction(self, data: Dict[str, Any]) -> None:
        conn = self._get_connection()
        cursor = conn.cursor()
        metadata = data.get("metadata")
        cursor.execute("""
            INSERT INTO contradictions
            (ledger_id, timestamp, old_memory_id, new_memory_id, drift_mean,
             drift_reason, confidence_delta, status, contradiction_type,
             affects_slots, query, summary, metadata, thread_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["ledger_id"], data["timestamp"],
            data["old_memory_id"], data["new_memory_id"],
            data["drift_mean"], data.get("drift_reason"),
            data.get("confidence_delta", 0.0),
            data["status"], data.get("contradiction_type", "conflict"),
            data.get("affects_slots"), data.get("query"),
            data.get("summary"),
            json.dumps(metadata) if metadata else None,
            data.get("thread_id"),
        ))
        conn.commit()
        conn.close()

    def get_contradictions(
        self, *, status=None, thread_id=None, memory_id=None, limit=100,
    ) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()

        conditions = []
        params: list = []

        if status is not None:
            conditions.append("status = ?")
            params.append(status)
        if thread_id is not None:
            conditions.append("COALESCE(thread_id, 'default') = ?")
            params.append(str(thread_id))
        if memory_id is not None:
            conditions.append("(old_memory_id = ? OR new_memory_id = ?)")
            params.extend([memory_id, memory_id])

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        cursor.execute(f"""
            SELECT ledger_id, timestamp, old_memory_id, new_memory_id,
                   drift_mean, drift_reason, confidence_delta, status,
                   contradiction_type, affects_slots, query, summary,
                   resolution_timestamp, resolution_method, merged_memory_id,
                   metadata, thread_id, lifecycle_state, confirmation_count,
                   disclosure_count, settled_at, archived_at
            FROM contradictions {where}
            ORDER BY timestamp DESC LIMIT ?
        """, params + [limit])

        columns = [
            "ledger_id", "timestamp", "old_memory_id", "new_memory_id",
            "drift_mean", "drift_reason", "confidence_delta", "status",
            "contradiction_type", "affects_slots", "query", "summary",
            "resolution_timestamp", "resolution_method", "merged_memory_id",
            "metadata", "thread_id", "lifecycle_state", "confirmation_count",
            "disclosure_count", "settled_at", "archived_at",
        ]
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            d = {}
            for i, col in enumerate(columns):
                if i < len(row):
                    d[col] = row[i]
            results.append(d)
        return results

    def update_contradiction(self, ledger_id: str, updates: Dict[str, Any]) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()
        sets = []
        params: list = []
        for k, v in updates.items():
            sets.append(f"{k} = ?")
            params.append(v)
        params.append(ledger_id)
        cursor.execute(
            f"UPDATE contradictions SET {', '.join(sets)} WHERE ledger_id = ?",
            params,
        )
        conn.commit()
        ok = cursor.rowcount > 0
        conn.close()
        return ok

    def insert_reflection(self, data: Dict[str, Any]) -> None:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO reflection_queue (timestamp, ledger_id, volatility, priority, context_json)
            VALUES (?, ?, ?, ?, ?)
        """, (
            data["timestamp"], data["ledger_id"], data["volatility"],
            data["priority"], json.dumps(data.get("context")) if data.get("context") else None,
        ))
        conn.commit()
        conn.close()

    def get_reflections(self, *, processed=False, priority=None) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        conditions = [f"processed = {int(processed)}"]
        params: list = []
        if priority:
            conditions.append("priority = ?")
            params.append(priority)

        where = f"WHERE {' AND '.join(conditions)}"
        cursor.execute(f"""
            SELECT queue_id, timestamp, ledger_id, volatility, priority, context_json
            FROM reflection_queue {where}
            ORDER BY
                CASE priority WHEN 'high' THEN 1 WHEN 'medium' THEN 2 WHEN 'low' THEN 3 END,
                volatility DESC, timestamp ASC
        """, params)
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "queue_id": r[0], "timestamp": r[1], "ledger_id": r[2],
                "volatility": r[3], "priority": r[4],
                "context": json.loads(r[5]) if r[5] else None,
            }
            for r in rows
        ]

    def mark_reflection_processed(self, queue_id: Any) -> None:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE reflection_queue SET processed = 1 WHERE queue_id = ?", (queue_id,))
        conn.commit()
        conn.close()

    def upsert_worklog_asked(self, ledger_id: str, ts: float) -> None:
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

    def upsert_worklog_answer(self, ledger_id: str, answer: str, ts: float) -> None:
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

    def count_contradictions(self, *, since, status=None, thread_id=None) -> int:
        conn = self._get_connection()
        cursor = conn.cursor()
        conds = ["timestamp > ?"]
        params: list = [since]
        if status:
            conds.append("status = ?")
            params.append(status)
        if thread_id:
            conds.append("COALESCE(thread_id, 'default') = ?")
            params.append(str(thread_id))
        cursor.execute(f"SELECT COUNT(*) FROM contradictions WHERE {' AND '.join(conds)}", params)
        val = cursor.fetchone()[0]
        conn.close()
        return val

    def avg_drift(self, *, since, thread_id=None) -> float:
        conn = self._get_connection()
        cursor = conn.cursor()
        conds = ["timestamp > ?"]
        params: list = [since]
        if thread_id:
            conds.append("COALESCE(thread_id, 'default') = ?")
            params.append(str(thread_id))
        cursor.execute(f"SELECT AVG(drift_mean) FROM contradictions WHERE {' AND '.join(conds)}", params)
        val = cursor.fetchone()[0] or 0.0
        conn.close()
        return val

    def count_pending_reflections(self) -> int:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM reflection_queue WHERE processed = 0")
        val = cursor.fetchone()[0]
        conn.close()
        return val


# ============================================================================
# In-Memory Backend
# ============================================================================


class InMemoryBackend:
    """Zero-dependency in-memory storage for testing and ephemeral use.

    All data lives in Python dicts/lists. No files, no SQLite, no cleanup.

    Usage::

        from groundcheck.backends import InMemoryBackend
        from groundcheck.ledger import ContradictionLedger

        ledger = ContradictionLedger(backend=InMemoryBackend())
    """

    def __init__(self):
        self._contradictions: Dict[str, Dict[str, Any]] = {}
        self._reflections: List[Dict[str, Any]] = []
        self._next_queue_id = 1
        self._worklog: Dict[str, Dict[str, Any]] = {}

    def init_storage(self) -> None:
        pass  # Nothing to initialize

    def insert_contradiction(self, data: Dict[str, Any]) -> None:
        self._contradictions[data["ledger_id"]] = dict(data)

    def get_contradictions(
        self, *, status=None, thread_id=None, memory_id=None, limit=100,
    ) -> List[Dict[str, Any]]:
        results = []
        for d in self._contradictions.values():
            if status is not None and d.get("status") != status:
                continue
            if thread_id is not None:
                entry_thread = d.get("thread_id") or "default"
                if entry_thread != str(thread_id):
                    continue
            if memory_id is not None:
                if d.get("old_memory_id") != memory_id and d.get("new_memory_id") != memory_id:
                    continue
            results.append(dict(d))

        results.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return results[:limit]

    def update_contradiction(self, ledger_id: str, updates: Dict[str, Any]) -> bool:
        if ledger_id not in self._contradictions:
            return False
        self._contradictions[ledger_id].update(updates)
        return True

    def insert_reflection(self, data: Dict[str, Any]) -> None:
        data = dict(data)
        data["queue_id"] = self._next_queue_id
        data["processed"] = False
        self._next_queue_id += 1
        self._reflections.append(data)

    def get_reflections(self, *, processed=False, priority=None) -> List[Dict[str, Any]]:
        results = []
        for r in self._reflections:
            if r.get("processed", False) != processed:
                continue
            if priority is not None and r.get("priority") != priority:
                continue
            results.append(dict(r))

        prio_order = {"high": 1, "medium": 2, "low": 3}
        results.sort(key=lambda x: (
            prio_order.get(x.get("priority", "low"), 9),
            -x.get("volatility", 0),
            x.get("timestamp", 0),
        ))
        return results

    def mark_reflection_processed(self, queue_id: Any) -> None:
        for r in self._reflections:
            if r.get("queue_id") == queue_id:
                r["processed"] = True
                break

    def upsert_worklog_asked(self, ledger_id: str, ts: float) -> None:
        if ledger_id in self._worklog:
            self._worklog[ledger_id]["last_asked_at"] = ts
            self._worklog[ledger_id]["ask_count"] = self._worklog[ledger_id].get("ask_count", 0) + 1
        else:
            self._worklog[ledger_id] = {
                "first_asked_at": ts, "last_asked_at": ts, "ask_count": 1,
                "last_user_answer": None, "last_user_answer_at": None,
            }

    def upsert_worklog_answer(self, ledger_id: str, answer: str, ts: float) -> None:
        if ledger_id not in self._worklog:
            self._worklog[ledger_id] = {
                "first_asked_at": None, "last_asked_at": None, "ask_count": 0,
            }
        self._worklog[ledger_id]["last_user_answer"] = (answer or "").strip()
        self._worklog[ledger_id]["last_user_answer_at"] = ts

    def count_contradictions(self, *, since, status=None, thread_id=None) -> int:
        count = 0
        for d in self._contradictions.values():
            if d.get("timestamp", 0) <= since:
                continue
            if status and d.get("status") != status:
                continue
            if thread_id:
                entry_thread = d.get("thread_id") or "default"
                if entry_thread != str(thread_id):
                    continue
            count += 1
        return count

    def avg_drift(self, *, since, thread_id=None) -> float:
        drifts = []
        for d in self._contradictions.values():
            if d.get("timestamp", 0) <= since:
                continue
            if thread_id:
                entry_thread = d.get("thread_id") or "default"
                if entry_thread != str(thread_id):
                    continue
            drifts.append(d.get("drift_mean", 0.0))
        return sum(drifts) / len(drifts) if drifts else 0.0

    def count_pending_reflections(self) -> int:
        return sum(1 for r in self._reflections if not r.get("processed", False))
