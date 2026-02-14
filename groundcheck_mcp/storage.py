"""SQLite-backed persistent memory storage for GroundCheck MCP server."""

import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

from groundcheck import Memory


class MemoryStore:
    """Persistent memory storage backed by SQLite.
    
    Stores facts as Memory objects with trust scores, timestamps,
    thread-level and namespace-level isolation.
    
    Namespaces allow project-scoped memory:
        - ``"global"`` — user-level facts visible across all projects
        - ``"my-production-app"`` — project-specific conventions
        - ``"playground"`` — a separate experimental project
    """
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_db()
    
    def _init_db(self) -> None:
        """Create tables if they don't exist, and migrate schema."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL DEFAULT 'default',
                text TEXT NOT NULL,
                trust REAL NOT NULL DEFAULT 0.7,
                source TEXT NOT NULL DEFAULT 'user',
                timestamp INTEGER NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_memories_thread 
                ON memories(thread_id);
            CREATE INDEX IF NOT EXISTS idx_memories_text 
                ON memories(text);
        """)
        self._conn.commit()
        self._migrate_add_namespace()
    
    def _migrate_add_namespace(self) -> None:
        """Add namespace column if it doesn't exist (v0.4 → v0.5 migration)."""
        cursor = self._conn.execute("PRAGMA table_info(memories)")
        columns = {row["name"] for row in cursor.fetchall()}
        if "namespace" not in columns:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace)"
            )
            self._conn.commit()
    
    def store(
        self,
        text: str,
        thread_id: str = "default",
        source: str = "user",
        trust: Optional[float] = None,
        metadata: Optional[Dict] = None,
        namespace: str = "default",
    ) -> Memory:
        """Store a new memory and return it as a Memory object.
        
        Args:
            text: The fact text to store.
            thread_id: Conversation-level isolation key.
            source: Origin of the fact (user/document/code/inferred).
            trust: Override trust score. Defaults by source.
            metadata: Arbitrary JSON-serializable metadata.
            namespace: Project-level scope. Use ``"global"`` for
                user-level facts that should be visible everywhere.
        
        Trust defaults:
            user: 0.70
            document: 0.60
            code: 0.80
            inferred: 0.40
        """
        trust_defaults = {
            "user": 0.70,
            "document": 0.60,
            "code": 0.80,
            "inferred": 0.40,
        }
        if trust is None:
            trust = trust_defaults.get(source, 0.50)
        
        ts = int(time.time())
        mem_id = f"mem_{namespace}_{thread_id}_{ts}_{hash(text) % 10000:04d}"
        
        meta_json = json.dumps(metadata) if metadata else None
        
        self._conn.execute(
            """INSERT INTO memories
               (id, thread_id, text, trust, source, timestamp, metadata, namespace)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (mem_id, thread_id, text, trust, source, ts, meta_json, namespace),
        )
        self._conn.commit()
        
        return Memory(
            id=mem_id,
            text=text,
            trust=trust,
            timestamp=ts,
            metadata={"source": source, "namespace": namespace, **(metadata or {})},
        )
    
    def query(
        self,
        query: str,
        thread_id: str = "default",
        limit: int = 50,
        namespace: str = "default",
        include_global: bool = True,
    ) -> List[Memory]:
        """Retrieve memories for a given thread and namespace.
        
        Args:
            query: Search hint (currently unused — returns all for thread).
            thread_id: Conversation-level isolation key.
            limit: Maximum number of memories to return.
            namespace: Project-level scope to query.
            include_global: If True (default), also include memories
                from the ``"global"`` namespace so user-level facts
                are always available regardless of which project is active.
        """
        if include_global and namespace != "global":
            rows = self._conn.execute(
                """SELECT id, text, trust, timestamp, metadata, namespace
                   FROM memories
                   WHERE thread_id = ? AND namespace IN (?, 'global')
                   ORDER BY trust DESC, timestamp DESC
                   LIMIT ?""",
                (thread_id, namespace, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT id, text, trust, timestamp, metadata, namespace
                   FROM memories
                   WHERE thread_id = ? AND namespace = ?
                   ORDER BY trust DESC, timestamp DESC
                   LIMIT ?""",
                (thread_id, namespace, limit),
            ).fetchall()
        
        memories = []
        for row in rows:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            meta = meta or {}
            meta["namespace"] = row["namespace"]
            memories.append(
                Memory(
                    id=row["id"],
                    text=row["text"],
                    trust=row["trust"],
                    timestamp=row["timestamp"],
                    metadata=meta,
                )
            )
        return memories
    
    def get_all(
        self,
        thread_id: str = "default",
        namespace: str = "default",
        include_global: bool = True,
    ) -> List[Memory]:
        """Get all memories for a thread within a namespace.
        
        When *include_global* is True (the default), memories stored in
        the ``"global"`` namespace are merged in so user-level facts like
        name, preferences, etc. are always available.
        """
        return self.query(
            "", thread_id=thread_id, limit=1000,
            namespace=namespace, include_global=include_global,
        )
    
    def update_trust(self, memory_id: str, new_trust: float) -> None:
        """Update the trust score for a specific memory."""
        self._conn.execute(
            "UPDATE memories SET trust = ? WHERE id = ?",
            (new_trust, memory_id),
        )
        self._conn.commit()
    
    def delete(self, memory_id: str) -> None:
        """Delete a specific memory."""
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()
    
    def clear_thread(self, thread_id: str, namespace: Optional[str] = None) -> int:
        """Delete all memories for a thread. Optionally scope to a namespace.
        
        If *namespace* is given, only memories in that namespace are deleted.
        Otherwise all namespaces for the thread are cleared.
        """
        if namespace is not None:
            cursor = self._conn.execute(
                "DELETE FROM memories WHERE thread_id = ? AND namespace = ?",
                (thread_id, namespace),
            )
        else:
            cursor = self._conn.execute(
                "DELETE FROM memories WHERE thread_id = ?", (thread_id,)
            )
        self._conn.commit()
        return cursor.rowcount
    
    def clear_namespace(self, namespace: str) -> int:
        """Delete all memories in a namespace (across all threads)."""
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE namespace = ?", (namespace,)
        )
        self._conn.commit()
        return cursor.rowcount
    
    def list_namespaces(self) -> List[str]:
        """Return all distinct namespaces that contain at least one memory."""
        rows = self._conn.execute(
            "SELECT DISTINCT namespace FROM memories ORDER BY namespace"
        ).fetchall()
        return [row["namespace"] for row in rows]
    
    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
