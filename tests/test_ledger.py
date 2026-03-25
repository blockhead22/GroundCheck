"""Tests for groundcheck.ledger — contradiction ledger."""

import os
import tempfile
import time
import pytest
from groundcheck.ledger import (
    ContradictionLedger,
    ContradictionEntry,
    ContradictionStatus,
    ContradictionType,
)


@pytest.fixture
def tmp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test_ledger.db")


@pytest.fixture
def ledger(tmp_db):
    return ContradictionLedger(db_path=tmp_db)


class TestContradictionEntry:
    def test_to_dict_from_dict_roundtrip(self):
        entry = ContradictionEntry(
            ledger_id="c1",
            timestamp=time.time(),
            old_memory_id="m1",
            new_memory_id="m2",
            drift_mean=0.75,
            status=ContradictionStatus.OPEN,
            contradiction_type=ContradictionType.CONFLICT,
            affects_slots="employer",
        )
        data = entry.to_dict()
        restored = ContradictionEntry.from_dict(data)
        assert restored.ledger_id == "c1"
        assert restored.drift_mean == 0.75
        assert restored.contradiction_type == ContradictionType.CONFLICT


class TestContradictionLedger:
    def test_record_and_retrieve(self, ledger):
        entry = ledger.record_contradiction(
            old_memory_id="m1",
            new_memory_id="m2",
            drift_mean=0.75,
            confidence_delta=0.2,
            old_text="works at Google",
            new_text="works at Microsoft",
        )
        assert entry.ledger_id.startswith("contra_")
        assert entry.status == ContradictionStatus.OPEN

        # Retrieve
        open_cs = ledger.get_open_contradictions()
        assert len(open_cs) >= 1
        assert open_cs[0].ledger_id == entry.ledger_id

    def test_resolve_updates_status(self, ledger):
        entry = ledger.record_contradiction(
            old_memory_id="m1",
            new_memory_id="m2",
            drift_mean=0.5,
            confidence_delta=0.1,
        )
        ledger.resolve_contradiction(entry.ledger_id, "deprecate_old")

        open_cs = ledger.get_open_contradictions()
        assert len(open_cs) == 0

        all_cs = ledger.get_all_contradictions()
        resolved = [c for c in all_cs if c.ledger_id == entry.ledger_id]
        assert len(resolved) == 1
        assert resolved[0].status == ContradictionStatus.RESOLVED

    def test_get_contradiction_by_memory(self, ledger):
        ledger.record_contradiction("m1", "m2", 0.5, 0.1)
        ledger.record_contradiction("m3", "m4", 0.6, 0.2)

        results = ledger.get_contradiction_by_memory("m1")
        assert len(results) == 1

    def test_has_open_contradiction(self, ledger):
        entry = ledger.record_contradiction("m1", "m2", 0.5, 0.1)
        assert ledger.has_open_contradiction("m1")
        ledger.resolve_contradiction(entry.ledger_id, "accept_both")
        assert not ledger.has_open_contradiction("m1")

    def test_stats(self, ledger):
        ledger.record_contradiction("m1", "m2", 0.5, 0.1)
        ledger.record_contradiction("m3", "m4", 0.7, 0.3)

        stats = ledger.get_contradiction_stats(days=1)
        assert stats["total_contradictions"] == 2
        assert stats["open"] == 2

    def test_reflection_queue(self, ledger):
        entry = ledger.record_contradiction("m1", "m2", 0.5, 0.1)
        ledger.queue_reflection(entry.ledger_id, 0.8)

        queue = ledger.get_reflection_queue()
        assert len(queue) == 1
        assert queue[0]["priority"] == "high"

        ledger.mark_reflection_processed(queue[0]["queue_id"])
        assert len(ledger.get_reflection_queue()) == 0

    def test_worklog(self, ledger):
        entry = ledger.record_contradiction("m1", "m2", 0.5, 0.1)
        ledger.mark_contradiction_asked(entry.ledger_id)
        ledger.record_contradiction_user_answer(entry.ledger_id, "I work at Microsoft")

    def test_thread_scoping(self, ledger):
        ledger.default_thread_id = "thread_a"
        ledger.record_contradiction("m1", "m2", 0.5, 0.1)

        ledger.default_thread_id = "thread_b"
        ledger.record_contradiction("m3", "m4", 0.6, 0.2)

        a_cs = ledger.get_open_contradictions(thread_id="thread_a")
        b_cs = ledger.get_open_contradictions(thread_id="thread_b")
        assert len(a_cs) == 1
        assert len(b_cs) == 1

    def test_classification_heuristics(self, ledger):
        entry = ledger.record_contradiction(
            old_memory_id="m1",
            new_memory_id="m2",
            drift_mean=0.5,
            confidence_delta=0.1,
            old_text="I work at Google",
            new_text="Actually I work at Microsoft",
        )
        assert entry.contradiction_type == ContradictionType.REVISION

    def test_summary_generation(self, ledger):
        entry = ledger.record_contradiction(
            old_memory_id="m1",
            new_memory_id="m2",
            drift_mean=0.6,
            confidence_delta=0.4,
        )
        assert entry.summary is not None
        assert "divergence" in entry.summary
