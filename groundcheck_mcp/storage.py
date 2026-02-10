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
    and thread-level isolation.
    """
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_db()
    
    def _init_db(self) -> None:
        """Create tables if they don't exist."""
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
    
    def store(
        self,
        text: str,
        thread_id: str = "default",
        source: str = "user",
        trust: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ) -> Memory:
        """Store a new memory and return it as a Memory object.
        
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
        mem_id = f"mem_{thread_id}_{ts}_{hash(text) % 10000:04d}"
        
        meta_json = json.dumps(metadata) if metadata else None
        
        self._conn.execute(
            """INSERT INTO memories (id, thread_id, text, trust, source, timestamp, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (mem_id, thread_id, text, trust, source, ts, meta_json),
        )
        self._conn.commit()
        
        return Memory(
            id=mem_id,
            text=text,
            trust=trust,
            timestamp=ts,
            metadata={"source": source, **(metadata or {})},
        )
    
    def query(
        self,
        query: str,
        thread_id: str = "default",
        limit: int = 50,
    ) -> List[Memory]:
        """Retrieve memories matching a query for a given thread.
        
        Uses simple substring matching on the text field.
        Returns all memories for the thread if query is broad.
        """
        # Get all thread memories, ordered by trust desc then recency desc
        rows = self._conn.execute(
            """SELECT id, text, trust, timestamp, metadata
               FROM memories
               WHERE thread_id = ?
               ORDER BY trust DESC, timestamp DESC
               LIMIT ?""",
            (thread_id, limit),
        ).fetchall()
        
        memories = []
        for row in rows:
            meta = json.loads(row["metadata"]) if row["metadata"] else None
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
    
    def get_all(self, thread_id: str = "default") -> List[Memory]:
        """Get all memories for a thread."""
        return self.query("", thread_id=thread_id, limit=1000)
    
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
    
    def clear_thread(self, thread_id: str) -> int:
        """Delete all memories for a thread. Returns count deleted."""
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE thread_id = ?", (thread_id,)
        )
        self._conn.commit()
        return cursor.rowcount
    
    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
