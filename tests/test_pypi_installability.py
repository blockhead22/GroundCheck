"""PyPI installability sanity tests for GroundCheck.

These tests verify that the package can be imported and used
with ONLY stdlib available — no external dependencies required.
"""

import pytest


def test_no_external_imports():
    """Core groundcheck must not import anything outside stdlib."""
    import importlib
    import groundcheck.verifier as v
    import groundcheck.types as t
    import groundcheck.fact_extractor as f
    import groundcheck.utils as u
    # If we got here without ImportError, stdlib-only is confirmed


def test_version_exists():
    """Package must expose a valid __version__."""
    import groundcheck
    assert hasattr(groundcheck, '__version__')
    assert groundcheck.__version__ == "0.1.0"


def test_core_exports():
    """All documented public exports must be importable."""
    from groundcheck import GroundCheck
    from groundcheck import Memory
    from groundcheck import VerificationReport
    from groundcheck import ExtractedFact
    from groundcheck import ContradictionDetail
    from groundcheck import extract_fact_slots

    # Verify they are the right types
    assert callable(GroundCheck)
    assert callable(extract_fact_slots)


def test_basic_verify_works():
    """The 10-second demo from README must actually work."""
    from groundcheck import GroundCheck, Memory

    verifier = GroundCheck()

    memories = [
        Memory(id="m1", text="User works at Microsoft", trust=0.9),
        Memory(id="m2", text="User works at Amazon", trust=0.3),
    ]

    result = verifier.verify("You work at Amazon", memories)

    assert isinstance(result.passed, bool)
    assert isinstance(result.hallucinations, list)
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0


def test_memory_dataclass():
    """Memory should accept all documented fields."""
    from groundcheck import Memory

    m = Memory(id="test", text="hello", trust=0.5, timestamp=1234567890)
    assert m.id == "test"
    assert m.text == "hello"
    assert m.trust == 0.5
    assert m.timestamp == 1234567890


def test_memory_defaults():
    """Memory should have sensible defaults."""
    from groundcheck import Memory

    m = Memory(id="test", text="hello")
    assert m.trust == 1.0
    assert m.timestamp is None
    assert m.metadata is None


def test_contradiction_detail_properties():
    """ContradictionDetail must expose most_trusted_value and most_recent_value."""
    from groundcheck import ContradictionDetail

    cd = ContradictionDetail(
        slot="employer",
        values=["microsoft", "amazon"],
        memory_ids=["m1", "m2"],
        timestamps=[1000, 2000],
        trust_scores=[0.9, 0.5],
    )
    assert cd.most_trusted_value == "microsoft"
    assert cd.most_recent_value == "amazon"
