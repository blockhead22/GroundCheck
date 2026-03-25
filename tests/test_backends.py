"""Tests for storage backends — InMemoryBackend and SQLiteBackend."""

import os
import tempfile
import time
import pytest
from groundcheck.backends import InMemoryBackend, SQLiteBackend, LedgerBackend
from groundcheck.ledger import (
    ContradictionLedger,
    ContradictionEntry,
    ContradictionStatus,
    ContradictionType,
)


# ============================================================================
# InMemoryBackend unit tests
# ============================================================================

class TestInMemoryBackend:
    def test_satisfies_protocol(self):
        backend = InMemoryBackend()
        assert isinstance(backend, LedgerBackend)

    def test_insert_and_get(self):
        backend = InMemoryBackend()
        backend.init_storage()
        backend.insert_contradiction({
            "ledger_id": "c1",
            "timestamp": time.time(),
            "old_memory_id": "m1",
            "new_memory_id": "m2",
            "drift_mean": 0.7,
            "status": "open",
        })
        rows = backend.get_contradictions()
        assert len(rows) == 1
        assert rows[0]["ledger_id"] == "c1"

    def test_filter_by_status(self):
        backend = InMemoryBackend()
        backend.init_storage()
        backend.insert_contradiction({
            "ledger_id": "c1", "timestamp": time.time(),
            "old_memory_id": "m1", "new_memory_id": "m2",
            "drift_mean": 0.5, "status": "open",
        })
        backend.insert_contradiction({
            "ledger_id": "c2", "timestamp": time.time(),
            "old_memory_id": "m3", "new_memory_id": "m4",
            "drift_mean": 0.6, "status": "resolved",
        })
        open_rows = backend.get_contradictions(status="open")
        assert len(open_rows) == 1
        assert open_rows[0]["ledger_id"] == "c1"

    def test_filter_by_thread(self):
        backend = InMemoryBackend()
        backend.init_storage()
        backend.insert_contradiction({
            "ledger_id": "c1", "timestamp": time.time(),
            "old_memory_id": "m1", "new_memory_id": "m2",
            "drift_mean": 0.5, "status": "open", "thread_id": "thread_a",
        })
        backend.insert_contradiction({
            "ledger_id": "c2", "timestamp": time.time(),
            "old_memory_id": "m3", "new_memory_id": "m4",
            "drift_mean": 0.6, "status": "open", "thread_id": "thread_b",
        })
        rows_a = backend.get_contradictions(thread_id="thread_a")
        assert len(rows_a) == 1
        assert rows_a[0]["ledger_id"] == "c1"

    def test_filter_by_memory_id(self):
        backend = InMemoryBackend()
        backend.init_storage()
        backend.insert_contradiction({
            "ledger_id": "c1", "timestamp": time.time(),
            "old_memory_id": "m1", "new_memory_id": "m2",
            "drift_mean": 0.5, "status": "open",
        })
        backend.insert_contradiction({
            "ledger_id": "c2", "timestamp": time.time(),
            "old_memory_id": "m3", "new_memory_id": "m4",
            "drift_mean": 0.6, "status": "open",
        })
        rows = backend.get_contradictions(memory_id="m1")
        assert len(rows) == 1

    def test_update_contradiction(self):
        backend = InMemoryBackend()
        backend.init_storage()
        backend.insert_contradiction({
            "ledger_id": "c1", "timestamp": time.time(),
            "old_memory_id": "m1", "new_memory_id": "m2",
            "drift_mean": 0.5, "status": "open",
        })
        ok = backend.update_contradiction("c1", {"status": "resolved"})
        assert ok
        rows = backend.get_contradictions(status="resolved")
        assert len(rows) == 1

    def test_update_nonexistent_returns_false(self):
        backend = InMemoryBackend()
        backend.init_storage()
        assert not backend.update_contradiction("nope", {"status": "resolved"})

    def test_reflection_queue(self):
        backend = InMemoryBackend()
        backend.init_storage()
        backend.insert_reflection({
            "timestamp": time.time(), "ledger_id": "c1",
            "volatility": 0.8, "priority": "high",
        })
        queue = backend.get_reflections()
        assert len(queue) == 1
        assert queue[0]["priority"] == "high"

        backend.mark_reflection_processed(queue[0]["queue_id"])
        assert len(backend.get_reflections()) == 0

    def test_worklog(self):
        backend = InMemoryBackend()
        backend.init_storage()
        backend.upsert_worklog_asked("c1", time.time())
        backend.upsert_worklog_asked("c1", time.time())
        assert backend._worklog["c1"]["ask_count"] == 2

        backend.upsert_worklog_answer("c1", "I work at Microsoft", time.time())
        assert backend._worklog["c1"]["last_user_answer"] == "I work at Microsoft"

    def test_count_and_avg(self):
        backend = InMemoryBackend()
        backend.init_storage()
        now = time.time()
        backend.insert_contradiction({
            "ledger_id": "c1", "timestamp": now,
            "old_memory_id": "m1", "new_memory_id": "m2",
            "drift_mean": 0.5, "status": "open",
        })
        backend.insert_contradiction({
            "ledger_id": "c2", "timestamp": now,
            "old_memory_id": "m3", "new_memory_id": "m4",
            "drift_mean": 0.7, "status": "open",
        })
        assert backend.count_contradictions(since=now - 1) == 2
        assert backend.count_contradictions(since=now - 1, status="open") == 2
        assert abs(backend.avg_drift(since=now - 1) - 0.6) < 0.01

    def test_newest_first_ordering(self):
        backend = InMemoryBackend()
        backend.init_storage()
        backend.insert_contradiction({
            "ledger_id": "c_old", "timestamp": time.time() - 100,
            "old_memory_id": "m1", "new_memory_id": "m2",
            "drift_mean": 0.5, "status": "open",
        })
        backend.insert_contradiction({
            "ledger_id": "c_new", "timestamp": time.time(),
            "old_memory_id": "m3", "new_memory_id": "m4",
            "drift_mean": 0.6, "status": "open",
        })
        rows = backend.get_contradictions()
        assert rows[0]["ledger_id"] == "c_new"


# ============================================================================
# SQLiteBackend satisfies protocol
# ============================================================================

class TestSQLiteBackend:
    def test_satisfies_protocol(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(db_path=os.path.join(tmpdir, "test.db"))
            assert isinstance(backend, LedgerBackend)


# ============================================================================
# Ledger integration with InMemoryBackend
# ============================================================================

class TestLedgerWithInMemoryBackend:
    def test_record_and_retrieve(self):
        ledger = ContradictionLedger(backend=InMemoryBackend())
        entry = ledger.record_contradiction(
            old_memory_id="m1", new_memory_id="m2",
            drift_mean=0.75, confidence_delta=0.2,
            old_text="works at Google", new_text="works at Microsoft",
        )
        assert entry.ledger_id.startswith("contra_")
        assert entry.status == ContradictionStatus.OPEN

        open_cs = ledger.get_open_contradictions()
        assert len(open_cs) >= 1

    def test_resolve(self):
        ledger = ContradictionLedger(backend=InMemoryBackend())
        entry = ledger.record_contradiction("m1", "m2", 0.5, 0.1)
        ledger.resolve_contradiction(entry.ledger_id, "deprecate_old")

        assert len(ledger.get_open_contradictions()) == 0
        all_cs = ledger.get_all_contradictions()
        assert any(c.status == ContradictionStatus.RESOLVED for c in all_cs)

    def test_by_memory(self):
        ledger = ContradictionLedger(backend=InMemoryBackend())
        ledger.record_contradiction("m1", "m2", 0.5, 0.1)
        ledger.record_contradiction("m3", "m4", 0.6, 0.2)

        results = ledger.get_contradiction_by_memory("m1")
        assert len(results) == 1

    def test_has_open(self):
        ledger = ContradictionLedger(backend=InMemoryBackend())
        entry = ledger.record_contradiction("m1", "m2", 0.5, 0.1)
        assert ledger.has_open_contradiction("m1")
        ledger.resolve_contradiction(entry.ledger_id, "accept_both")
        assert not ledger.has_open_contradiction("m1")

    def test_stats(self):
        ledger = ContradictionLedger(backend=InMemoryBackend())
        ledger.record_contradiction("m1", "m2", 0.5, 0.1)
        ledger.record_contradiction("m3", "m4", 0.7, 0.3)

        stats = ledger.get_contradiction_stats(days=1)
        assert stats["total_contradictions"] == 2
        assert stats["open"] == 2

    def test_reflection_queue(self):
        ledger = ContradictionLedger(backend=InMemoryBackend())
        entry = ledger.record_contradiction("m1", "m2", 0.5, 0.1)
        ledger.queue_reflection(entry.ledger_id, 0.8)

        queue = ledger.get_reflection_queue()
        assert len(queue) == 1
        assert queue[0]["priority"] == "high"

        ledger.mark_reflection_processed(queue[0]["queue_id"])
        assert len(ledger.get_reflection_queue()) == 0

    def test_worklog(self):
        ledger = ContradictionLedger(backend=InMemoryBackend())
        entry = ledger.record_contradiction("m1", "m2", 0.5, 0.1)
        ledger.mark_contradiction_asked(entry.ledger_id)
        ledger.record_contradiction_user_answer(entry.ledger_id, "I work at Microsoft")

    def test_thread_scoping(self):
        ledger = ContradictionLedger(backend=InMemoryBackend())
        ledger.default_thread_id = "thread_a"
        ledger.record_contradiction("m1", "m2", 0.5, 0.1)

        ledger.default_thread_id = "thread_b"
        ledger.record_contradiction("m3", "m4", 0.6, 0.2)

        a_cs = ledger.get_open_contradictions(thread_id="thread_a")
        b_cs = ledger.get_open_contradictions(thread_id="thread_b")
        assert len(a_cs) == 1
        assert len(b_cs) == 1

    def test_backend_property(self):
        backend = InMemoryBackend()
        ledger = ContradictionLedger(backend=backend)
        assert ledger.backend is backend

    def test_summary_generation(self):
        ledger = ContradictionLedger(backend=InMemoryBackend())
        entry = ledger.record_contradiction("m1", "m2", 0.6, 0.4)
        assert entry.summary is not None
        assert "divergence" in entry.summary

    def test_classification_heuristics(self):
        ledger = ContradictionLedger(backend=InMemoryBackend())
        entry = ledger.record_contradiction(
            old_memory_id="m1", new_memory_id="m2",
            drift_mean=0.5, confidence_delta=0.1,
            old_text="I work at Google",
            new_text="Actually I work at Microsoft",
        )
        assert entry.contradiction_type == ContradictionType.REVISION
