"""Integration tests for GroundCheck."""

import pytest
from groundcheck import GroundCheck, Memory


def test_end_to_end_verification():
    """Test complete end-to-end verification flow."""
    verifier = GroundCheck()
    
    # Setup memories
    memories = [
        Memory(id="m1", text="User works at Microsoft", trust=0.9),
        Memory(id="m2", text="User lives in Seattle", trust=0.85),
        Memory(id="m3", text="User's name is Alice", trust=0.95)
    ]
    
    # Test fully grounded text
    result = verifier.verify(
        "Your name is Alice, you work at Microsoft, and you live in Seattle",
        memories
    )
    
    assert result.passed == True
    assert len(result.hallucinations) == 0
    assert result.confidence > 0.8
    assert len(result.grounding_map) >= 2


def test_end_to_end_with_hallucinations():
    """Test end-to-end with partial hallucinations."""
    verifier = GroundCheck()
    
    memories = [
        Memory(id="m1", text="User works at Microsoft"),
        Memory(id="m2", text="User lives in Seattle")
    ]
    
    # Mixed grounded and hallucinated claims
    result = verifier.verify(
        "You work at Amazon and live in Seattle and your name is Bob",
        memories,
        mode="strict"
    )
    
    assert result.passed == False
    assert "Amazon" in result.hallucinations
    assert "Bob" in result.hallucinations
    assert "Seattle" not in result.hallucinations
    assert result.corrected is not None


def test_memory_claim_sanitization():
    """Test that memory claims are properly sanitized."""
    verifier = GroundCheck()
    
    memories = [
        Memory(id="m1", text="User works at Microsoft")
    ]
    
    # Text with explicit memory claim
    result = verifier.verify(
        "I remember you work at Amazon",
        memories,
        mode="strict"
    )
    
    assert result.passed == False
    assert "Amazon" in result.hallucinations
    # Corrected text should not contain the memory claim line
    assert result.corrected is not None
    assert "remember" not in result.corrected.lower()


def test_high_trust_vs_low_trust():
    """Test that high trust memories are preferred."""
    verifier = GroundCheck()
    
    # Conflicting memories with different trust scores
    memories = [
        Memory(id="m1", text="User works at Microsoft", trust=0.3),
        Memory(id="m2", text="User works at Amazon", trust=0.95)
    ]
    
    # Verify claim that matches high-trust memory
    result = verifier.verify("You work at Amazon", memories)
    
    assert result.passed == True
    assert result.confidence > 0.8


def test_multiple_fact_types():
    """Test verification with multiple different fact types."""
    verifier = GroundCheck()
    
    memories = [
        Memory(id="m1", text="My name is Alice"),
        Memory(id="m2", text="I work at Microsoft"),
        Memory(id="m3", text="I live in Seattle"),
        Memory(id="m4", text="My favorite color is blue")
    ]
    
    result = verifier.verify(
        "Your name is Alice, you work at Microsoft, live in Seattle, and like blue",
        memories
    )
    
    # Should successfully ground most claims
    assert len(result.grounding_map) >= 2
    assert result.confidence > 0.5


def test_no_memories_provided():
    """Test behavior when no memories are provided."""
    verifier = GroundCheck()
    
    result = verifier.verify(
        "You work at Microsoft and live in Seattle",
        []
    )
    
    assert result.passed == False
    assert len(result.hallucinations) > 0
    assert result.confidence == 0.0


def test_correction_replaces_values():
    """Test that correction properly replaces hallucinated values."""
    verifier = GroundCheck()
    
    memories = [
        Memory(id="m1", text="User works at Microsoft")
    ]
    
    result = verifier.verify(
        "You work at Amazon",
        memories,
        mode="strict"
    )
    
    assert result.corrected is not None
    assert "Amazon" not in result.corrected
    assert "Microsoft" in result.corrected


def test_structured_vs_natural_facts():
    """Test that both structured and natural language facts work."""
    verifier = GroundCheck()
    
    memories = [
        Memory(id="m1", text="FACT: name = Alice"),  # Structured
        Memory(id="m2", text="I work at Microsoft")   # Natural
    ]
    
    result = verifier.verify(
        "Your name is Alice and you work at Microsoft",
        memories
    )
    
    assert result.passed == True
    assert len(result.grounding_map) >= 2


def test_case_insensitive_matching():
    """Test that matching is case-insensitive."""
    verifier = GroundCheck()
    
    memories = [
        Memory(id="m1", text="User works at microsoft")  # lowercase
    ]
    
    result = verifier.verify(
        "You work at Microsoft",  # TitleCase
        memories
    )
    
    assert result.passed == True


def test_verification_report_completeness():
    """Test that VerificationReport contains all expected fields."""
    verifier = GroundCheck()
    
    memories = [
        Memory(id="m1", text="User works at Microsoft")
    ]
    
    result = verifier.verify(
        "You work at Amazon and live in Seattle",
        memories,
        mode="strict"
    )
    
    # Check all fields are present
    assert hasattr(result, 'original')
    assert hasattr(result, 'corrected')
    assert hasattr(result, 'passed')
    assert hasattr(result, 'hallucinations')
    assert hasattr(result, 'grounding_map')
    assert hasattr(result, 'confidence')
    assert hasattr(result, 'facts_extracted')
    assert hasattr(result, 'facts_supported')
    
    # Check types
    assert isinstance(result.original, str)
    assert isinstance(result.passed, bool)
    assert isinstance(result.hallucinations, list)
    assert isinstance(result.grounding_map, dict)
    assert isinstance(result.confidence, float)
