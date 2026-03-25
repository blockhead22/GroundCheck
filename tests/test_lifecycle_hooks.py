"""Tests for lifecycle event hooks (#15)."""

import time
import pytest
from groundcheck.lifecycle import (
    ContradictionLifecycleState,
    ContradictionLifecycleEntry,
    ContradictionLifecycle,
    LifecycleHook,
)


class TestLifecycleHooks:
    def test_hook_fires_on_transition(self):
        events = []

        def hook(entry, old_state, new_state):
            events.append((entry.ledger_id, old_state, new_state))

        lc = ContradictionLifecycle(hooks=[hook])
        entry = ContradictionLifecycleEntry(ledger_id="c1", confirmation_count=1)
        lc.record_confirmation(entry)
        # 2 confirmations -> SETTLING
        assert entry.state == ContradictionLifecycleState.SETTLING
        assert len(events) == 1
        assert events[0] == ("c1", ContradictionLifecycleState.ACTIVE, ContradictionLifecycleState.SETTLING)

    def test_hook_not_fired_when_no_transition(self):
        events = []

        def hook(entry, old_state, new_state):
            events.append((entry.ledger_id, old_state, new_state))

        lc = ContradictionLifecycle(hooks=[hook])
        entry = ContradictionLifecycleEntry(ledger_id="c1", confirmation_count=0)
        lc.record_confirmation(entry)
        # 1 confirmation, needs 2 -> no transition
        assert entry.state == ContradictionLifecycleState.ACTIVE
        assert len(events) == 0

    def test_on_state_change_registers_hook(self):
        events = []
        lc = ContradictionLifecycle()
        lc.on_state_change(lambda e, o, n: events.append(n))

        entry = ContradictionLifecycleEntry(ledger_id="c1", confirmation_count=1)
        lc.record_confirmation(entry)
        assert len(events) == 1
        assert events[0] == ContradictionLifecycleState.SETTLING

    def test_remove_hook(self):
        events = []

        def hook(entry, old_state, new_state):
            events.append(new_state)

        lc = ContradictionLifecycle()
        lc.on_state_change(hook)
        lc.remove_hook(hook)

        entry = ContradictionLifecycleEntry(ledger_id="c1", confirmation_count=1)
        lc.record_confirmation(entry)
        assert len(events) == 0

    def test_multiple_hooks_all_fire(self):
        events_a = []
        events_b = []

        lc = ContradictionLifecycle()
        lc.on_state_change(lambda e, o, n: events_a.append(n))
        lc.on_state_change(lambda e, o, n: events_b.append(n))

        entry = ContradictionLifecycleEntry(ledger_id="c1", confirmation_count=1)
        lc.record_confirmation(entry)
        assert len(events_a) == 1
        assert len(events_b) == 1

    def test_hook_error_does_not_break_transition(self):
        """A failing hook should not prevent the state transition."""
        def bad_hook(entry, old_state, new_state):
            raise RuntimeError("hook failed")

        events = []

        def good_hook(entry, old_state, new_state):
            events.append(new_state)

        lc = ContradictionLifecycle(hooks=[bad_hook, good_hook])
        entry = ContradictionLifecycleEntry(ledger_id="c1", confirmation_count=1)
        lc.record_confirmation(entry)
        # State should still transition despite bad_hook
        assert entry.state == ContradictionLifecycleState.SETTLING
        # Good hook should still have fired
        assert len(events) == 1

    def test_transition_method_fires_hooks(self):
        """The explicit transition() method should also fire hooks."""
        events = []

        lc = ContradictionLifecycle()
        lc.on_state_change(lambda e, o, n: events.append((o, n)))

        entry = ContradictionLifecycleEntry(
            ledger_id="c1",
            detected_at=time.time() - 8 * 86400,  # 8 days old -> past freshness
        )
        new_state = lc.transition(entry)
        assert new_state == ContradictionLifecycleState.SETTLING
        assert entry.state == ContradictionLifecycleState.SETTLING
        assert len(events) == 1
        assert events[0] == (ContradictionLifecycleState.ACTIVE, ContradictionLifecycleState.SETTLING)

    def test_transition_no_change_no_hook(self):
        events = []
        lc = ContradictionLifecycle()
        lc.on_state_change(lambda e, o, n: events.append(n))

        entry = ContradictionLifecycleEntry(ledger_id="c1")  # Fresh, no confirmations
        new_state = lc.transition(entry)
        assert new_state == ContradictionLifecycleState.ACTIVE
        assert len(events) == 0

    def test_full_lifecycle_with_hooks(self):
        """Walk entry through ACTIVE -> SETTLING -> SETTLED with hooks at each step."""
        transitions = []

        def hook(entry, old, new):
            transitions.append(f"{old.value}->{new.value}")

        lc = ContradictionLifecycle(hooks=[hook])
        entry = ContradictionLifecycleEntry(ledger_id="c1")

        # 2 confirmations -> SETTLING
        lc.record_confirmation(entry)
        lc.record_confirmation(entry)
        assert entry.state == ContradictionLifecycleState.SETTLING

        # 3 more confirmations -> SETTLED (total 5)
        lc.record_confirmation(entry)
        lc.record_confirmation(entry)
        lc.record_confirmation(entry)
        assert entry.state == ContradictionLifecycleState.SETTLED

        assert transitions == ["active->settling", "settling->settled"]

    def test_constructor_hooks_list(self):
        """Hooks passed to constructor should be separate from other instances."""
        events = []
        hook = lambda e, o, n: events.append(n)
        lc1 = ContradictionLifecycle(hooks=[hook])
        lc2 = ContradictionLifecycle()

        entry = ContradictionLifecycleEntry(ledger_id="c1", confirmation_count=1)
        lc2.record_confirmation(entry)
        # lc2 has no hooks, so events should be empty
        assert len(events) == 0
