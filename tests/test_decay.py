"""Tests for groundcheck.decay — trust decay and reinforcement."""

import os
import sqlite3
import tempfile
import time
import pytest
from groundcheck.decay import (
    _compute_exponential_decay,
    _compute_drift_aware_boost,
    run_trust_decay_pass,
    reinforce_memory,
    TRUST_FLOOR,
    TRUST_CEILING,
    GRACE_PERIOD_DAYS,
)


def _create_test_db(path):
    """Create a minimal memory DB for testing."""
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            text TEXT,
            trust REAL DEFAULT 0.85,
            timestamp REAL
        )
    """)
    return conn


class TestExponentialDecay:
    def test_decay_reduces_trust(self):
        age_30_days = 30 * 86400
        new_trust = _compute_exponential_decay(age_30_days, 0.85)
        assert new_trust < 0.85

    def test_trust_floor_respected(self):
        age_365_days = 365 * 86400
        new_trust = _compute_exponential_decay(age_365_days, 0.25)
        assert new_trust >= TRUST_FLOOR

    def test_young_memory_barely_decays(self):
        age_1_day = 86400
        new_trust = _compute_exponential_decay(age_1_day, 0.85)
        assert new_trust > 0.84  # Very small decay


class TestDriftAwareBoost:
    def test_boost_increases_trust(self):
        new_trust = _compute_drift_aware_boost(0.7, "works at Google", "at Google", is_correction=False)
        assert new_trust > 0.7

    def test_correction_boost_larger(self):
        regular = _compute_drift_aware_boost(0.7, "works at Google", "at Google", is_correction=False)
        correction = _compute_drift_aware_boost(0.7, "works at Google", "at Google", is_correction=True)
        assert correction >= regular

    def test_ceiling_respected(self):
        new_trust = _compute_drift_aware_boost(0.94, "test", "test", is_correction=True)
        assert new_trust <= TRUST_CEILING


class TestDecayPass:
    def test_decay_pass_on_test_db(self):
        import groundcheck.decay as mod
        mod._last_decay_ts = 0  # Reset throttle

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            conn = _create_test_db(db_path)

            # Insert old memory (past grace period)
            old_ts = time.time() - (GRACE_PERIOD_DAYS + 1) * 86400
            conn.execute(
                "INSERT INTO memories (id, text, trust, timestamp) VALUES (?, ?, ?, ?)",
                ("m1", "User works at Google", 0.85, old_ts),
            )
            # Insert young memory (within grace period)
            conn.execute(
                "INSERT INTO memories (id, text, trust, timestamp) VALUES (?, ?, ?, ?)",
                ("m2", "User lives in Seattle", 0.85, time.time()),
            )
            conn.commit()
            conn.close()

            result = run_trust_decay_pass(db_path=db_path)
            assert not result["skipped"]
            assert result["decayed"] == 1  # Only old memory decayed
            assert result["total_stale_checked"] == 1

    def test_decay_pass_returns_dict(self):
        """Decay pass always returns a dict, regardless of DB state."""
        import groundcheck.decay as mod
        mod._last_decay_ts = 0
        result = run_trust_decay_pass(db_path="Z:/definitely_nonexistent_dir/path.db")
        assert isinstance(result, dict)
        # It either found a fallback DB or skipped — both are valid
        assert "skipped" in result or "decayed" in result


class TestReinforceMemory:
    def test_reinforce_increases_trust(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            conn = _create_test_db(db_path)
            conn.execute(
                "INSERT INTO memories (id, text, trust, timestamp) VALUES (?, ?, ?, ?)",
                ("m1", "User works at Google", 0.7, time.time()),
            )
            conn.commit()
            conn.close()

            result = reinforce_memory("m1", db_path=db_path)
            assert result is True

            # Verify trust increased
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT trust FROM memories WHERE id = ?", ("m1",)).fetchone()
            assert row["trust"] > 0.7
            conn.close()

    def test_reinforce_missing_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            _create_test_db(db_path).close()
            result = reinforce_memory("nonexistent", db_path=db_path)
            assert result is False
