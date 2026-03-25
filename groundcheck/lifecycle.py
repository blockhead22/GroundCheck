"""
Contradiction Lifecycle and Disclosure Policy.

Implements:
1. Extended contradiction states (Active -> Settling -> Settled -> Archived)
2. State transition logic based on user confirmations and time
3. Disclosure policy with "disclosure budgets" to reduce noise
4. User transparency preferences

Design Philosophy:
- Contradictions have a lifecycle, not just binary resolution
- Not all contradictions need immediate disclosure
- User preferences guide transparency level
- High-stakes domains always get disclosure
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ContradictionLifecycleState(str, Enum):
    """Extended contradiction states for lifecycle management.

    Lifecycle flow: ACTIVE -> SETTLING -> SETTLED -> ARCHIVED
    """
    ACTIVE = "active"
    SETTLING = "settling"
    SETTLED = "settled"
    ARCHIVED = "archived"


@dataclass
class ContradictionLifecycleEntry:
    """Extended contradiction entry with lifecycle tracking."""
    ledger_id: str
    state: ContradictionLifecycleState = ContradictionLifecycleState.ACTIVE

    # Timestamps
    detected_at: float = field(default_factory=time.time)
    settled_at: Optional[float] = None
    archived_at: Optional[float] = None

    # User interaction tracking
    confirmation_count: int = 0
    disclosure_count: int = 0
    last_mentioned: float = field(default_factory=time.time)

    # Affected facts
    affected_slots: Set[str] = field(default_factory=set)
    old_value: Optional[str] = None
    new_value: Optional[str] = None

    # Configuration
    freshness_window: float = 7 * 86400  # 7 days in seconds

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "ledger_id": self.ledger_id,
            "state": self.state.value,
            "detected_at": self.detected_at,
            "settled_at": self.settled_at,
            "archived_at": self.archived_at,
            "confirmation_count": self.confirmation_count,
            "disclosure_count": self.disclosure_count,
            "last_mentioned": self.last_mentioned,
            "affected_slots": list(self.affected_slots),
            "old_value": self.old_value,
            "new_value": self.new_value,
            "freshness_window": self.freshness_window,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ContradictionLifecycleEntry:
        """Create from dictionary."""
        return cls(
            ledger_id=data["ledger_id"],
            state=ContradictionLifecycleState(data.get("state", "active")),
            detected_at=data.get("detected_at", time.time()),
            settled_at=data.get("settled_at"),
            archived_at=data.get("archived_at"),
            confirmation_count=data.get("confirmation_count", 0),
            disclosure_count=data.get("disclosure_count", 0),
            last_mentioned=data.get("last_mentioned", time.time()),
            affected_slots=set(data.get("affected_slots", [])),
            old_value=data.get("old_value"),
            new_value=data.get("new_value"),
            freshness_window=data.get("freshness_window", 7 * 86400),
        )

    @property
    def age_seconds(self) -> float:
        return time.time() - self.detected_at

    @property
    def age_days(self) -> float:
        return self.age_seconds / 86400

    @property
    def is_stale(self) -> bool:
        return self.age_seconds > self.freshness_window


# Type alias for lifecycle event callbacks
LifecycleHook = Callable[
    ["ContradictionLifecycleEntry", "ContradictionLifecycleState", "ContradictionLifecycleState"],
    None,
]


class ContradictionLifecycle:
    """Manages contradiction state transitions with pluggable event hooks.

    Transition rules:
    1. ACTIVE -> SETTLING: 2+ confirmations OR past freshness window
    2. SETTLING -> SETTLED: 5+ confirmations OR past 2x freshness window
    3. SETTLED -> ARCHIVED: After 30 days

    Event hooks:
        Register callbacks via ``on_state_change(callback)`` or pass a list
        to the constructor. Callbacks receive ``(entry, old_state, new_state)``.

    Example::

        lc = ContradictionLifecycle()
        lc.on_state_change(lambda entry, old, new: print(f"{entry.ledger_id}: {old} -> {new}"))
    """

    ACTIVE_TO_SETTLING_CONFIRMATIONS = 2
    SETTLING_TO_SETTLED_CONFIRMATIONS = 5
    ARCHIVE_AFTER_DAYS = 30

    def __init__(
        self,
        active_to_settling_confirmations: int = 2,
        settling_to_settled_confirmations: int = 5,
        archive_after_days: int = 30,
        hooks: Optional[List[LifecycleHook]] = None,
    ):
        self.active_to_settling = active_to_settling_confirmations
        self.settling_to_settled = settling_to_settled_confirmations
        self.archive_days = archive_after_days
        self._hooks: List[LifecycleHook] = list(hooks or [])

    def on_state_change(self, callback: LifecycleHook) -> None:
        """Register a callback for state transitions.

        Callback signature: ``(entry, old_state, new_state) -> None``.
        """
        self._hooks.append(callback)

    def remove_hook(self, callback: LifecycleHook) -> None:
        """Remove a previously registered hook."""
        self._hooks = [h for h in self._hooks if h is not callback]

    def _fire_hooks(
        self,
        entry: ContradictionLifecycleEntry,
        old_state: ContradictionLifecycleState,
        new_state: ContradictionLifecycleState,
    ) -> None:
        for hook in self._hooks:
            try:
                hook(entry, old_state, new_state)
            except Exception as e:
                logger.warning("Lifecycle hook error: %s", e)

    def update_state(
        self, entry: ContradictionLifecycleEntry,
    ) -> ContradictionLifecycleState:
        """Evaluate and return the appropriate state for a contradiction."""
        now = time.time()
        age = now - entry.detected_at
        current_state = entry.state

        if current_state == ContradictionLifecycleState.ACTIVE:
            if entry.confirmation_count >= self.active_to_settling:
                return ContradictionLifecycleState.SETTLING
            if age > entry.freshness_window:
                return ContradictionLifecycleState.SETTLING

        if current_state == ContradictionLifecycleState.SETTLING:
            if entry.confirmation_count >= self.settling_to_settled:
                entry.settled_at = now
                return ContradictionLifecycleState.SETTLED
            if age > entry.freshness_window * 2:
                entry.settled_at = now
                return ContradictionLifecycleState.SETTLED

        if current_state == ContradictionLifecycleState.SETTLED:
            archive_threshold = self.archive_days * 86400
            if age > archive_threshold:
                entry.archived_at = now
                return ContradictionLifecycleState.ARCHIVED

        return current_state

    def transition(
        self, entry: ContradictionLifecycleEntry,
    ) -> ContradictionLifecycleState:
        """Evaluate state and fire hooks if a transition occurs.

        This is the preferred method for external callers — it both
        updates the entry's state and notifies all registered hooks.

        Returns the (possibly new) state.
        """
        old_state = entry.state
        new_state = self.update_state(entry)
        if new_state != old_state:
            entry.state = new_state
            self._fire_hooks(entry, old_state, new_state)
        return new_state

    def record_confirmation(
        self, entry: ContradictionLifecycleEntry,
    ) -> ContradictionLifecycleState:
        """Record a user confirmation and update state if needed."""
        old_state = entry.state
        entry.confirmation_count += 1
        entry.last_mentioned = time.time()
        new_state = self.update_state(entry)
        if new_state != old_state:
            entry.state = new_state
            self._fire_hooks(entry, old_state, new_state)
        else:
            entry.state = new_state
        return new_state

    def record_disclosure(
        self, entry: ContradictionLifecycleEntry,
    ) -> None:
        """Record that a contradiction was disclosed to the user."""
        entry.disclosure_count += 1
        entry.last_mentioned = time.time()


class TransparencyLevel(str, Enum):
    """User preference for transparency/disclosure level."""
    MINIMAL = "minimal"
    BALANCED = "balanced"
    AUDIT_HEAVY = "audit_heavy"


class MemoryStyle(str, Enum):
    """User preference for how memories are handled."""
    STICKY = "sticky"
    NORMAL = "normal"
    FORGETFUL = "forgetful"


@dataclass
class UserTransparencyPrefs:
    """User preferences for transparency and disclosure."""
    transparency_level: TransparencyLevel = TransparencyLevel.BALANCED
    memory_style: MemoryStyle = MemoryStyle.NORMAL
    always_disclose_domains: Set[str] = field(default_factory=lambda: {
        "medical", "financial", "legal",
    })
    never_nag_domains: Set[str] = field(default_factory=set)
    max_disclosures_per_session: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transparency_level": self.transparency_level.value,
            "memory_style": self.memory_style.value,
            "always_disclose_domains": list(self.always_disclose_domains),
            "never_nag_domains": list(self.never_nag_domains),
            "max_disclosures_per_session": self.max_disclosures_per_session,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UserTransparencyPrefs:
        return cls(
            transparency_level=TransparencyLevel(
                data.get("transparency_level", "balanced")
            ),
            memory_style=MemoryStyle(data.get("memory_style", "normal")),
            always_disclose_domains=set(data.get("always_disclose_domains", [
                "medical", "financial", "legal",
            ])),
            never_nag_domains=set(data.get("never_nag_domains", [])),
            max_disclosures_per_session=data.get("max_disclosures_per_session", 3),
        )


class DisclosurePolicy:
    """Determines when contradictions should be disclosed to the user.

    Balances transparency with user experience. Not every contradiction
    needs immediate disclosure.
    """

    HIGH_STAKES_ATTRIBUTES: Set[str] = {
        "medical_diagnosis", "medication", "allergy", "blood_type", "medical_condition",
        "account_balance", "account_number", "credit_score", "salary", "income",
        "legal_status", "citizenship", "visa_status",
        "emergency_contact", "address", "phone_number",
    }

    def __init__(
        self,
        user_prefs: Optional[UserTransparencyPrefs] = None,
        lifecycle: Optional[ContradictionLifecycle] = None,
    ):
        self.user_prefs = user_prefs or UserTransparencyPrefs()
        self.lifecycle = lifecycle or ContradictionLifecycle()
        self._session_disclosures = 0

    def reset_session(self) -> None:
        self._session_disclosures = 0

    def should_disclose(
        self,
        contradiction: ContradictionLifecycleEntry,
        query_context: str = "",
        force_check_query: bool = False,
    ) -> bool:
        """Determine if a contradiction should be disclosed."""
        # Always disclose conditions
        if query_context and self._is_direct_query(query_context, contradiction):
            return True
        if contradiction.state == ContradictionLifecycleState.ACTIVE:
            return True
        if self._is_high_stakes(contradiction):
            return True
        if self.user_prefs.transparency_level == TransparencyLevel.AUDIT_HEAVY:
            return True

        # Skip disclosure conditions
        if contradiction.state == ContradictionLifecycleState.ARCHIVED:
            return False
        if self._session_disclosures >= self.user_prefs.max_disclosures_per_session:
            return False
        if self.user_prefs.transparency_level == TransparencyLevel.MINIMAL:
            return False
        if contradiction.disclosure_count >= 3:
            return False

        # Balanced: SETTLING state
        if contradiction.state == ContradictionLifecycleState.SETTLING:
            if contradiction.confirmation_count >= 2:
                return False
            return True

        # SETTLED state — only on direct query
        return force_check_query and self._is_direct_query(query_context, contradiction)

    def _is_direct_query(
        self, query: str, contradiction: ContradictionLifecycleEntry,
    ) -> bool:
        if not query:
            return False
        query_lower = query.lower()
        for slot in contradiction.affected_slots:
            slot_normalized = slot.lower().replace("_", " ")
            if slot_normalized in query_lower or slot in query_lower:
                return True
        if contradiction.old_value and contradiction.old_value.lower() in query_lower:
            return True
        if contradiction.new_value and contradiction.new_value.lower() in query_lower:
            return True
        return False

    def _is_high_stakes(self, contradiction: ContradictionLifecycleEntry) -> bool:
        if contradiction.affected_slots & self.HIGH_STAKES_ATTRIBUTES:
            return True
        for slot in contradiction.affected_slots:
            for domain in self.user_prefs.always_disclose_domains:
                if domain in slot.lower():
                    return True
        return False

    def record_disclosure(
        self, contradiction: ContradictionLifecycleEntry,
    ) -> None:
        self._session_disclosures += 1
        self.lifecycle.record_disclosure(contradiction)

    def get_disclosure_priority(
        self,
        contradictions: List[ContradictionLifecycleEntry],
        query_context: str = "",
    ) -> List[ContradictionLifecycleEntry]:
        """Sort contradictions by disclosure priority (highest first)."""

        def priority_score(c: ContradictionLifecycleEntry) -> float:
            score = 0.0
            state_scores = {
                ContradictionLifecycleState.ACTIVE: 100,
                ContradictionLifecycleState.SETTLING: 50,
                ContradictionLifecycleState.SETTLED: 10,
                ContradictionLifecycleState.ARCHIVED: 0,
            }
            score += state_scores.get(c.state, 0)
            if self._is_high_stakes(c):
                score += 200
            if self._is_direct_query(query_context, c):
                score += 300
            age_days = c.age_days
            if age_days < 1:
                score += 50
            elif age_days < 7:
                score += 20
            score -= c.disclosure_count * 10
            return score

        return sorted(contradictions, key=priority_score, reverse=True)
