"""Trust weighting edge case tests for GroundCheck."""

import pytest
from groundcheck import GroundCheck, Memory


def test_trust_weighted_confidence():
    """Confidence should weight toward high-trust memories."""
    verifier = GroundCheck()
    memories = [
        Memory(id="m1", text="User works at Microsoft", trust=0.95),
        Memory(id="m2", text="User works at Amazon", trust=0.1),
    ]
    result = verifier.verify("You work at Microsoft", memories)
    assert result.passed is True
    assert result.confidence > 0.8


def test_equal_trust_requires_disclosure():
    """When trust scores are close and both high, both should be disclosed."""
    verifier = GroundCheck()
    memories = [
        Memory(id="m1", text="User works at Microsoft", trust=0.85),
        Memory(id="m2", text="User works at Amazon", trust=0.85),
    ]
    result = verifier.verify("You work at Microsoft", memories)
    assert result.requires_disclosure is True


def test_zero_trust_memory_ignored():
    """Memory with trust=0 should not affect verification outcome."""
    verifier = GroundCheck()
    memories = [
        Memory(id="m1", text="User works at Microsoft", trust=0.9),
        Memory(id="m2", text="User works at Amazon", trust=0.0),
    ]
    result = verifier.verify("You work at Microsoft", memories)
    assert result.passed is True


def test_single_memory_high_trust():
    """Single high-trust memory should give high confidence."""
    verifier = GroundCheck()
    memories = [
        Memory(id="m1", text="User works at Microsoft", trust=0.95),
    ]
    result = verifier.verify("You work at Microsoft", memories)
    assert result.passed is True
    assert result.confidence > 0.9


def test_all_low_trust_still_grounds():
    """Even low-trust memories can ground claims when they agree."""
    verifier = GroundCheck()
    memories = [
        Memory(id="m1", text="User works at Microsoft", trust=0.3),
        Memory(id="m2", text="User works at Microsoft", trust=0.4),
    ]
    result = verifier.verify("You work at Microsoft", memories)
    assert result.passed is True


def test_trust_difference_below_threshold_requires_disclosure():
    """Small trust difference with both memories credible requires disclosure."""
    verifier = GroundCheck()
    memories = [
        Memory(id="m1", text="User works at Microsoft", trust=0.90),
        Memory(id="m2", text="User works at Amazon", trust=0.85),
    ]
    result = verifier.verify("You work at Amazon", memories)
    # Trust difference is 0.05, both are credible (>=0.75)
    assert result.requires_disclosure is True


def test_trust_difference_above_threshold_no_disclosure():
    """Large trust difference means low-trust memory is noise — no disclosure needed."""
    verifier = GroundCheck()
    memories = [
        Memory(id="m1", text="User works at Microsoft", trust=0.2),
        Memory(id="m2", text="User works at Amazon", trust=0.95),
    ]
    result = verifier.verify("You work at Amazon", memories)
    assert result.passed is True
    assert result.requires_disclosure is False
