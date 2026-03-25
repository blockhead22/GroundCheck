"""Tests for groundcheck.lifecycle — contradiction lifecycle and disclosure policy."""

import time
import pytest
from groundcheck.lifecycle import (
    ContradictionLifecycleState,
    ContradictionLifecycleEntry,
    ContradictionLifecycle,
    TransparencyLevel,
    MemoryStyle,
    UserTransparencyPrefs,
    DisclosurePolicy,
)


class TestLifecycleEntry:
    def test_default_state_is_active(self):
        entry = ContradictionLifecycleEntry(ledger_id="c1")
        assert entry.state == ContradictionLifecycleState.ACTIVE

    def test_serialization_roundtrip(self):
        entry = ContradictionLifecycleEntry(
            ledger_id="c1",
            confirmation_count=3,
            affected_slots={"employer", "location"},
            old_value="Google",
            new_value="Microsoft",
        )
        data = entry.to_dict()
        restored = ContradictionLifecycleEntry.from_dict(data)
        assert restored.ledger_id == "c1"
        assert restored.confirmation_count == 3
        assert restored.affected_slots == {"employer", "location"}
        assert restored.old_value == "Google"
        assert restored.new_value == "Microsoft"

    def test_age_and_staleness(self):
        entry = ContradictionLifecycleEntry(
            ledger_id="c1",
            detected_at=time.time() - 8 * 86400,  # 8 days ago
        )
        assert entry.age_days >= 7.9
        assert entry.is_stale  # Default freshness is 7 days


class TestContradictionLifecycle:
    def test_active_to_settling_by_confirmations(self):
        lc = ContradictionLifecycle()
        entry = ContradictionLifecycleEntry(ledger_id="c1")
        entry.confirmation_count = 2
        new_state = lc.update_state(entry)
        assert new_state == ContradictionLifecycleState.SETTLING

    def test_active_to_settling_by_age(self):
        lc = ContradictionLifecycle()
        entry = ContradictionLifecycleEntry(
            ledger_id="c1",
            detected_at=time.time() - 8 * 86400,
        )
        new_state = lc.update_state(entry)
        assert new_state == ContradictionLifecycleState.SETTLING

    def test_settling_to_settled_by_confirmations(self):
        lc = ContradictionLifecycle()
        entry = ContradictionLifecycleEntry(
            ledger_id="c1",
            state=ContradictionLifecycleState.SETTLING,
            confirmation_count=5,
        )
        new_state = lc.update_state(entry)
        assert new_state == ContradictionLifecycleState.SETTLED

    def test_settled_to_archived_by_age(self):
        lc = ContradictionLifecycle()
        entry = ContradictionLifecycleEntry(
            ledger_id="c1",
            state=ContradictionLifecycleState.SETTLED,
            detected_at=time.time() - 31 * 86400,
        )
        new_state = lc.update_state(entry)
        assert new_state == ContradictionLifecycleState.ARCHIVED

    def test_record_confirmation_advances_state(self):
        lc = ContradictionLifecycle()
        entry = ContradictionLifecycleEntry(ledger_id="c1", confirmation_count=1)
        new_state = lc.record_confirmation(entry)
        assert entry.confirmation_count == 2
        assert new_state == ContradictionLifecycleState.SETTLING

    def test_record_disclosure_increments_count(self):
        lc = ContradictionLifecycle()
        entry = ContradictionLifecycleEntry(ledger_id="c1")
        lc.record_disclosure(entry)
        assert entry.disclosure_count == 1


class TestDisclosurePolicy:
    def test_active_always_disclosed(self):
        policy = DisclosurePolicy()
        entry = ContradictionLifecycleEntry(
            ledger_id="c1",
            state=ContradictionLifecycleState.ACTIVE,
        )
        assert policy.should_disclose(entry)

    def test_high_stakes_always_disclosed(self):
        policy = DisclosurePolicy()
        entry = ContradictionLifecycleEntry(
            ledger_id="c1",
            state=ContradictionLifecycleState.SETTLING,
            affected_slots={"medication"},
            confirmation_count=5,
        )
        assert policy.should_disclose(entry)

    def test_archived_not_disclosed(self):
        policy = DisclosurePolicy()
        entry = ContradictionLifecycleEntry(
            ledger_id="c1",
            state=ContradictionLifecycleState.ARCHIVED,
        )
        assert not policy.should_disclose(entry)

    def test_minimal_suppresses_non_critical(self):
        prefs = UserTransparencyPrefs(transparency_level=TransparencyLevel.MINIMAL)
        policy = DisclosurePolicy(user_prefs=prefs)
        entry = ContradictionLifecycleEntry(
            ledger_id="c1",
            state=ContradictionLifecycleState.SETTLING,
            confirmation_count=0,
        )
        assert not policy.should_disclose(entry)

    def test_audit_heavy_always_discloses(self):
        prefs = UserTransparencyPrefs(transparency_level=TransparencyLevel.AUDIT_HEAVY)
        policy = DisclosurePolicy(user_prefs=prefs)
        entry = ContradictionLifecycleEntry(
            ledger_id="c1",
            state=ContradictionLifecycleState.SETTLING,
        )
        assert policy.should_disclose(entry)

    def test_session_limit_respected(self):
        prefs = UserTransparencyPrefs(max_disclosures_per_session=1)
        policy = DisclosurePolicy(user_prefs=prefs)
        entry = ContradictionLifecycleEntry(
            ledger_id="c1",
            state=ContradictionLifecycleState.SETTLING,
        )
        policy.record_disclosure(entry)
        entry2 = ContradictionLifecycleEntry(
            ledger_id="c2",
            state=ContradictionLifecycleState.SETTLING,
        )
        assert not policy.should_disclose(entry2)

    def test_priority_ordering(self):
        policy = DisclosurePolicy()
        c1 = ContradictionLifecycleEntry(ledger_id="c1", state=ContradictionLifecycleState.ACTIVE)
        c2 = ContradictionLifecycleEntry(ledger_id="c2", state=ContradictionLifecycleState.ARCHIVED)
        c3 = ContradictionLifecycleEntry(
            ledger_id="c3",
            state=ContradictionLifecycleState.SETTLING,
            affected_slots={"medication"},
        )
        ordered = policy.get_disclosure_priority([c2, c1, c3])
        assert ordered[0].ledger_id == "c3"  # High stakes
        assert ordered[1].ledger_id == "c1"  # Active


class TestUserTransparencyPrefs:
    def test_serialization_roundtrip(self):
        prefs = UserTransparencyPrefs(
            transparency_level=TransparencyLevel.AUDIT_HEAVY,
            memory_style=MemoryStyle.FORGETFUL,
        )
        data = prefs.to_dict()
        restored = UserTransparencyPrefs.from_dict(data)
        assert restored.transparency_level == TransparencyLevel.AUDIT_HEAVY
        assert restored.memory_style == MemoryStyle.FORGETFUL
