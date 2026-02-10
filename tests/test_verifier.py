"""Core tests for GroundCheck verifier."""

import pytest
from groundcheck import GroundCheck, Memory


def test_basic_grounding_pass():
    """Test that correctly grounded text passes verification."""
    verifier = GroundCheck()
    memories = [Memory(id="m1", text="User works at Microsoft")]
    
    result = verifier.verify("You work at Microsoft", memories)
    
    assert result.passed == True
    assert len(result.hallucinations) == 0


def test_basic_grounding_fail():
    """Test that hallucinated claims are detected."""
    verifier = GroundCheck()
    memories = [Memory(id="m1", text="User works at Microsoft")]
    
    result = verifier.verify("You work at Amazon", memories)
    
    assert result.passed == False
    assert "Amazon" in result.hallucinations


def test_partial_grounding():
    """Test mixed grounded and ungrounded claims."""
    verifier = GroundCheck()
    memories = [
        Memory(id="m1", text="User works at Microsoft"),
        Memory(id="m2", text="User lives in Seattle")
    ]
    
    result = verifier.verify(
        "You work at Amazon and live in Seattle", 
        memories
    )
    
    assert result.passed == False
    assert "Amazon" in result.hallucinations
    assert "Seattle" not in result.hallucinations
    assert result.grounding_map.get("Seattle") == "m2"


def test_correction_mode():
    """Test that corrections are generated in strict mode."""
    verifier = GroundCheck()
    memories = [Memory(id="m1", text="User works at Microsoft")]
    
    result = verifier.verify(
        "You work at Amazon", 
        memories, 
        mode="strict"
    )
    
    assert result.corrected is not None
    assert "Microsoft" in result.corrected
    assert "Amazon" not in result.corrected


def test_fact_slot_extraction():
    """Test that fact slots are correctly extracted."""
    from groundcheck.fact_extractor import extract_fact_slots
    
    facts = extract_fact_slots("My name is Alice and I work at Microsoft")
    
    assert "name" in facts
    assert facts["name"].value == "Alice"
    assert "employer" in facts
    assert facts["employer"].value == "Microsoft"


def test_empty_memories():
    """Test behavior with no retrieved context."""
    verifier = GroundCheck()
    
    result = verifier.verify("You work at Microsoft", [])
    
    assert result.passed == False
    assert "Microsoft" in result.hallucinations


def test_confidence_scoring():
    """Test confidence scores are calculated."""
    verifier = GroundCheck()
    memories = [Memory(id="m1", text="User works at Microsoft", trust=0.9)]
    
    result = verifier.verify("You work at Microsoft", memories)
    
    assert result.confidence > 0.8


def test_paraphrase_detection():
    """Test that paraphrases are recognized as grounded.
    
    Note: This test uses simple string matching. More sophisticated
    semantic matching could be added as an optional feature.
    """
    verifier = GroundCheck()
    memories = [Memory(id="m1", text="I work at Microsoft")]
    
    result = verifier.verify("You work at Microsoft", memories)
    
    # Should recognize the fact even with different phrasing
    assert result.passed == True


def test_multiple_memory_support():
    """Test claim supported by multiple memories."""
    verifier = GroundCheck()
    memories = [
        Memory(id="m1", text="User works at Microsoft"),
        Memory(id="m2", text="I work at Microsoft")
    ]
    
    result = verifier.verify("You work at Microsoft", memories)
    
    assert result.passed == True
    assert result.grounding_map.get("Microsoft") in ["m1", "m2"]


def test_trust_weighted_verification():
    """Test that low-trust memories are handled appropriately."""
    verifier = GroundCheck()
    memories = [
        Memory(id="m1", text="User works at Microsoft", trust=0.2),
        Memory(id="m2", text="User works at Amazon", trust=0.9)
    ]
    
    result = verifier.verify("You work at Amazon", memories)
    
    # Should prefer high-trust memory
    assert result.confidence > 0.8


def test_structured_fact_format():
    """Test that structured FACT: format is recognized."""
    verifier = GroundCheck()
    memories = [Memory(id="m1", text="FACT: employer = Microsoft")]
    
    result = verifier.verify("You work at Microsoft", memories)
    
    assert result.passed == True


def test_permissive_mode():
    """Test that permissive mode doesn't generate corrections."""
    verifier = GroundCheck()
    memories = [Memory(id="m1", text="User works at Microsoft")]
    
    result = verifier.verify(
        "You work at Amazon",
        memories,
        mode="permissive"
    )
    
    assert result.passed == False
    assert "Amazon" in result.hallucinations
    assert result.corrected is None


def test_extract_claims():
    """Test the extract_claims method."""
    verifier = GroundCheck()
    
    claims = verifier.extract_claims("My name is Bob and I live in Denver")
    
    assert "name" in claims
    assert claims["name"].value == "Bob"
    assert "location" in claims
    assert claims["location"].value == "Denver"


def test_find_support():
    """Test the find_support method."""
    verifier = GroundCheck()
    memories = [
        Memory(id="m1", text="User works at Microsoft"),
        Memory(id="m2", text="User lives in Seattle")
    ]
    
    # Extract a claim
    claims = verifier.extract_claims("I work at Microsoft")
    employer_claim = claims["employer"]
    
    # Find supporting memory
    support = verifier.find_support(employer_claim, memories)
    
    assert support is not None
    assert support.id == "m1"


def test_build_grounding_map():
    """Test the build_grounding_map method."""
    verifier = GroundCheck()
    memories = [
        Memory(id="m1", text="User works at Microsoft"),
        Memory(id="m2", text="User lives in Seattle")
    ]
    
    claims = verifier.extract_claims("I work at Microsoft and live in Seattle")
    grounding_map = verifier.build_grounding_map(claims, memories)
    
    assert "Microsoft" in grounding_map
    assert "Seattle" in grounding_map
    assert grounding_map["Microsoft"] == "m1"
    assert grounding_map["Seattle"] == "m2"


def test_empty_text():
    """Test verification with empty text."""
    verifier = GroundCheck()
    memories = [Memory(id="m1", text="User works at Microsoft")]
    
    result = verifier.verify("", memories)
    
    assert result.passed == True
    assert len(result.hallucinations) == 0


def test_no_facts_extracted():
    """Test text with no extractable facts."""
    verifier = GroundCheck()
    memories = [Memory(id="m1", text="User works at Microsoft")]
    
    result = verifier.verify("Hello, how are you today?", memories)
    
    # No facts to verify, so should pass
    assert result.passed == True
    assert len(result.hallucinations) == 0


def test_compound_value_splitting():
    """Test that compound values are split and verified individually."""
    verifier = GroundCheck()
    memories = [
        Memory(id="m1", text="User knows Python"),
        Memory(id="m2", text="User knows JavaScript")
    ]
    
    # Test with all supported values
    result = verifier.verify("You use Python and JavaScript", memories)
    assert result.passed == True
    assert len(result.hallucinations) == 0
    assert "Python" in result.grounding_map
    assert "JavaScript" in result.grounding_map
    
    # Test with partially supported values
    result = verifier.verify("You use Python, JavaScript, Ruby, and Go", memories)
    assert result.passed == False
    assert "Ruby" in result.hallucinations
    assert "Go" in result.hallucinations
    assert "Python" not in result.hallucinations
    assert "JavaScript" not in result.hallucinations
    assert result.grounding_map.get("Python") == "m1"
    assert result.grounding_map.get("JavaScript") == "m2"


def test_paraphrase_fuzzy_matching():
    """Test that paraphrases are recognized via fuzzy matching."""
    verifier = GroundCheck()
    
    # Test "employed by" vs "work at"
    memories = [Memory(id="m1", text="User is employed by Microsoft")]
    result = verifier.verify("You work at Microsoft", memories)
    assert result.passed == True
    assert len(result.hallucinations) == 0
    
    # Test "resides in" vs "live in"
    memories = [Memory(id="m1", text="User resides in Seattle, Washington")]
    result = verifier.verify("You live in Seattle", memories)
    assert result.passed == True
    assert len(result.hallucinations) == 0
    
    # Test "Stanford University" vs "Stanford" with fuzzy matching
    memories = [Memory(id="m1", text="User graduated from Stanford University")]
    result = verifier.verify("You studied at Stanford", memories)
    assert "Stanford" not in result.hallucinations
    assert result.passed == True
    

def test_partial_grounding_with_details():
    """Test detection of hallucinated details in partially grounded statements."""
    verifier = GroundCheck()
    
    # Test location with extra details
    memories = [Memory(id="m1", text="User lives in Seattle")]
    result = verifier.verify("You live in Seattle", memories)
    assert result.passed == True
    
    # Test employer with wrong title
    memories = [
        Memory(id="m1", text="User works at Microsoft"),
        Memory(id="m2", text="User is a Software Engineer")
    ]
    result = verifier.verify("You work at Microsoft as a Product Manager", memories)
    assert result.passed == False
    assert "Product Manager" in result.hallucinations
    assert "Microsoft" not in result.hallucinations


def test_compound_splitting_various_separators():
    """Test splitting with different separators."""
    from groundcheck.fact_extractor import split_compound_values
    
    # Commas
    assert split_compound_values("Python, JavaScript, Ruby") == ["Python", "JavaScript", "Ruby"]
    
    # "and"
    assert split_compound_values("Python and JavaScript") == ["Python", "JavaScript"]
    
    # "or"
    assert split_compound_values("Python or JavaScript") == ["Python", "JavaScript"]
    
    # Slashes
    assert split_compound_values("Python/JavaScript") == ["Python", "JavaScript"]
    
    # Mixed (Oxford comma)
    assert split_compound_values("Python, JavaScript, and Ruby") == ["Python", "JavaScript", "Ruby"]
    
    # Semicolons
    assert split_compound_values("Python; JavaScript") == ["Python", "JavaScript"]
    
    # Single value (no splitting)
    assert split_compound_values("Python") == ["Python"]


def test_partial_grounding_accuracy():
    """Test partial grounding detection (some claims true, some false)."""
    verifier = GroundCheck()
    
    memories = [
        Memory(id="m1", text="User knows Python", trust=0.9),
        Memory(id="m2", text="User knows JavaScript", trust=0.9)
    ]
    
    # Test with all supported programming languages (should pass)
    result = verifier.verify("You use Python and JavaScript", memories)
    assert result.passed == True
    assert len(result.hallucinations) == 0
    
    # Test with partially supported programming languages (2 correct, 2 hallucinations)
    result = verifier.verify("You use Python, JavaScript, Ruby, and Go", memories)
    assert result.passed == False  # Should fail due to Ruby and Go
    assert "Ruby" in result.hallucinations
    assert "Go" in result.hallucinations
    # Ensure Python and JavaScript are NOT in hallucinations
    assert "Python" not in result.hallucinations
    assert "JavaScript" not in result.hallucinations


def test_semantic_paraphrase_matching():
    """Test that semantic paraphrases are correctly matched."""
    verifier = GroundCheck()
    
    # Only run if embedding model is available
    if not hasattr(verifier, 'embedding_model') or verifier.embedding_model is None:
        pytest.skip("Semantic matching not available (embedding model not loaded)")
    
    # Test employer paraphrases
    memories = [
        Memory(id="m1", text="User works at Google", trust=0.9)
    ]
    
    paraphrases = [
        "You are employed by Google",
        "You work for Google",
        "Your employer is Google",
    ]
    
    for paraphrase in paraphrases:
        result = verifier.verify(paraphrase, memories)
        assert result.passed, f"Should accept paraphrase: {paraphrase}"


def test_semantic_location_paraphrases():
    """Test location paraphrases."""
    verifier = GroundCheck()
    
    # Only run if embedding model is available
    if not hasattr(verifier, 'embedding_model') or verifier.embedding_model is None:
        pytest.skip("Semantic matching not available (embedding model not loaded)")
    
    memories = [
        Memory(id="m1", text="User lives in Seattle", trust=0.9)
    ]
    
    paraphrases = [
        "You reside in Seattle",
        "You are based in Seattle",
        "You are located in Seattle",
    ]
    
    for paraphrase in paraphrases:
        result = verifier.verify(paraphrase, memories)
        assert result.passed, f"Should accept paraphrase: {paraphrase}"


def test_semantic_threshold_prevents_false_positives():
    """Test that semantic threshold prevents false positives."""
    verifier = GroundCheck()
    
    # Only run if embedding model is available
    if not hasattr(verifier, 'embedding_model') or verifier.embedding_model is None:
        pytest.skip("Semantic matching not available (embedding model not loaded)")
    
    memories = [
        Memory(id="m1", text="User works at Google", trust=0.9)
    ]
    
    # Should NOT match (semantically different)
    false_matches = [
        "You work at Microsoft",  # Different company
    ]
    
    for text in false_matches:
        result = verifier.verify(text, memories)
        # These should fail (hallucination)
        assert result.passed == False, f"Should reject false match: {text}"
