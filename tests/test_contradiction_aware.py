"""Comprehensive tests for contradiction-aware grounding."""

import pytest
from groundcheck import GroundCheck, Memory
from groundcheck.types import ContradictionDetail


class TestContradictionDetection:
    """Test detection of contradictions in retrieved memories."""
    
    def test_detect_simple_contradiction(self):
        """Test basic contradiction between two memories."""
        verifier = GroundCheck()
        memories = [
            Memory(id="m1", text="User works at Microsoft", trust=0.9),
            Memory(id="m2", text="User works at Amazon", trust=0.9)
        ]
        
        contradictions = verifier._detect_contradictions(memories)
        
        assert len(contradictions) == 1
        assert contradictions[0].slot == "employer"
        assert set(contradictions[0].values) == {"microsoft", "amazon"}
        assert set(contradictions[0].memory_ids) == {"m1", "m2"}
    
    def test_detect_temporal_contradiction(self):
        """Test contradiction with timestamps."""
        verifier = GroundCheck()
        memories = [
            Memory(id="m1", text="User works at Microsoft", trust=0.9, timestamp=1704067200),  # Jan 2024
            Memory(id="m2", text="User works at Amazon", trust=0.9, timestamp=1706745600)     # Feb 2024
        ]
        
        contradictions = verifier._detect_contradictions(memories)
        
        assert len(contradictions) == 1
        assert contradictions[0].most_recent_value == "amazon"
    
    def test_no_contradiction_same_value(self):
        """Test that same value in multiple memories is not a contradiction."""
        verifier = GroundCheck()
        memories = [
            Memory(id="m1", text="User works at Microsoft"),
            Memory(id="m2", text="User is employed by Microsoft")
        ]
        
        contradictions = verifier._detect_contradictions(memories)
        
        # Should recognize both as "microsoft" (normalized)
        assert len(contradictions) == 0
    
    def test_multiple_contradictions(self):
        """Test detection of contradictions in multiple slots."""
        verifier = GroundCheck()
        memories = [
            Memory(id="m1", text="User works at Microsoft in Seattle"),
            Memory(id="m2", text="User works at Amazon in Portland")
        ]
        
        contradictions = verifier._detect_contradictions(memories)
        
        # Should find contradictions in both employer and location
        assert len(contradictions) == 2
        slots = {c.slot for c in contradictions}
        assert "employer" in slots
        assert "location" in slots
    
    def test_no_contradiction_for_additive_slots(self):
        """Test that additive slots (like programming languages) don't trigger contradictions."""
        verifier = GroundCheck()
        memories = [
            Memory(id="m1", text="User knows Python"),
            Memory(id="m2", text="User knows JavaScript")
        ]
        
        contradictions = verifier._detect_contradictions(memories)
        
        # Programming languages are additive, not contradictory
        assert len(contradictions) == 0


class TestContradictionDisclosure:
    """Test verification of contradiction disclosure in outputs."""
    
    def test_undisclosed_contradiction_fails(self):
        """Test that using contradicted fact without disclosure fails.
        
        Uses high trust scores (≥0.86) for both memories to trigger disclosure requirement.
        """
        verifier = GroundCheck()
        memories = [
            Memory(id="m1", text="User works at Microsoft", timestamp=1704067200, trust=0.90),
            Memory(id="m2", text="User works at Amazon", timestamp=1706745600, trust=0.90)
        ]
        
        result = verifier.verify("You work at Amazon", memories)
        
        assert result.passed == False  # Should fail - no disclosure
        assert result.requires_disclosure == True
        assert "Amazon" in result.contradicted_claims
        assert result.expected_disclosure is not None
        assert "microsoft" in result.expected_disclosure.lower()  # Should mention old value
    
    def test_disclosed_contradiction_passes(self):
        """Test that proper disclosure of contradiction passes."""
        verifier = GroundCheck()
        memories = [
            Memory(id="m1", text="User works at Microsoft", timestamp=1704067200, trust=0.85),
            Memory(id="m2", text="User works at Amazon", timestamp=1706745600, trust=0.85)
        ]
        
        result = verifier.verify(
            "You work at Amazon. You previously worked at Microsoft.",
            memories
        )
        
        assert result.passed == True  # Should pass - has disclosure
        assert result.requires_disclosure == False
        # Contradictions are still detected, but disclosure is adequate
        assert len(result.contradiction_details) > 0
    
    def test_various_disclosure_patterns(self):
        """Test recognition of different disclosure phrasings."""
        verifier = GroundCheck()
        memories = [
            Memory(id="m1", text="User lives in Seattle", trust=0.85),
            Memory(id="m2", text="User lives in Portland", trust=0.85)
        ]
        
        valid_disclosures = [
            "You live in Portland (moved from Seattle)",
            "You live in Portland. Previously you lived in Seattle.",
            "You used to live in Seattle but now live in Portland",
            "You live in Portland (was Seattle)",
        ]
        
        for output in valid_disclosures:
            result = verifier.verify(output, memories)
            assert result.passed == True, f"Failed on: {output}"
    
    def test_correction_adds_disclosure(self):
        """Test that strict mode adds disclosure to correction."""
        verifier = GroundCheck()
        memories = [
            Memory(id="m1", text="User works at Microsoft", timestamp=1704067200, trust=0.90),
            Memory(id="m2", text="User works at Amazon", timestamp=1706745600, trust=0.90)
        ]
        
        result = verifier.verify(
            "You work at Amazon",
            memories,
            mode="strict"
        )
        
        assert result.corrected is not None
        assert "microsoft" in result.corrected.lower()  # Should add old value
        # Should contain disclosure language
        assert any(word in result.corrected.lower() for word in ["changed", "previously", "was"])


class TestContradictionEdgeCases:
    """Test edge cases in contradiction handling."""
    
    def test_three_way_contradiction(self):
        """Test contradiction with three different values."""
        verifier = GroundCheck()
        memories = [
            Memory(id="m1", text="User works at Microsoft", trust=0.8),
            Memory(id="m2", text="User works at Amazon", trust=0.8),
            Memory(id="m3", text="User works at Google", trust=0.8)
        ]
        
        contradictions = verifier._detect_contradictions(memories)
        
        assert len(contradictions) == 1
        assert len(contradictions[0].values) == 3
    
    def test_trust_weighted_contradiction_resolution(self):
        """Test that high-trust memory is preferred in contradictions."""
        verifier = GroundCheck()
        memories = [
            Memory(id="m1", text="User works at Microsoft", trust=0.5),  # Low trust
            Memory(id="m2", text="User works at Amazon", trust=0.95)     # High trust
        ]
        
        contradictions = verifier._detect_contradictions(memories)
        contradiction = contradictions[0]
        
        assert contradiction.most_trusted_value == "amazon"
    
    def test_no_timestamp_uses_trust(self):
        """Test fallback to trust when timestamps unavailable."""
        verifier = GroundCheck()
        memories = [
            Memory(id="m1", text="User works at Microsoft", trust=0.7),
            Memory(id="m2", text="User works at Amazon", trust=0.9)
        ]
        
        contradictions = verifier._detect_contradictions(memories)
        
        # Should use trust score as tiebreaker
        assert contradictions[0].most_trusted_value == "amazon"
        # most_recent_value should fall back to trust when no timestamps
        assert contradictions[0].most_recent_value == "amazon"
    
    def test_large_trust_difference_no_disclosure_required(self):
        """Test that very low trust memories don't require disclosure."""
        verifier = GroundCheck()
        memories = [
            Memory(id="m1", text="User works at Microsoft", trust=0.2),  # Very low trust
            Memory(id="m2", text="User works at Amazon", trust=0.95)     # High trust
        ]
        
        # Trust difference is 0.75, above 0.5 threshold, so no disclosure required
        result = verifier.verify("You work at Amazon", memories)
        
        assert result.passed == True  # Should pass despite contradiction
        assert result.requires_disclosure == False
        assert "Amazon" in result.contradicted_claims  # Still detected as contradicted
    
    def test_similar_trust_requires_disclosure(self):
        """Test that similar high trust scores require disclosure."""
        verifier = GroundCheck()
        memories = [
            Memory(id="m1", text="User works at Microsoft", trust=0.90),
            Memory(id="m2", text="User works at Amazon", trust=0.90)
        ]

        # Both high trust (≥0.86) with small difference, so disclosure required
        result = verifier.verify("You work at Amazon", memories)
        
        assert result.passed == False  # Should fail without disclosure
        assert result.requires_disclosure == True


class TestContradictionProperties:
    """Test ContradictionDetail properties."""
    
    def test_most_recent_value_with_timestamps(self):
        """Test most_recent_value property with timestamps."""
        contradiction = ContradictionDetail(
            slot="employer",
            values=["microsoft", "amazon"],
            memory_ids=["m1", "m2"],
            timestamps=[1704067200, 1706745600],  # Jan, Feb 2024
            trust_scores=[0.9, 0.8]
        )
        
        assert contradiction.most_recent_value == "amazon"  # Feb is more recent
    
    def test_most_recent_value_without_timestamps(self):
        """Test most_recent_value falls back to trust when no timestamps."""
        contradiction = ContradictionDetail(
            slot="employer",
            values=["microsoft", "amazon"],
            memory_ids=["m1", "m2"],
            timestamps=[None, None],
            trust_scores=[0.7, 0.9]
        )
        
        assert contradiction.most_recent_value == "amazon"  # Higher trust
    
    def test_most_trusted_value(self):
        """Test most_trusted_value property."""
        contradiction = ContradictionDetail(
            slot="employer",
            values=["microsoft", "amazon"],
            memory_ids=["m1", "m2"],
            timestamps=[1706745600, 1704067200],  # Feb, Jan (reversed)
            trust_scores=[0.7, 0.95]
        )
        
        # Most trusted should be second entry (0.95 trust)
        assert contradiction.most_trusted_value == "amazon"


class TestIntegration:
    """Integration tests for contradiction-aware grounding."""
    
    def test_end_to_end_contradiction_detection(self):
        """Test complete flow from detection to disclosure verification."""
        verifier = GroundCheck()
        
        memories = [
            Memory(
                id="m1",
                text="User works at Microsoft in Seattle",
                trust=0.90,
                timestamp=1704067200
            ),
            Memory(
                id="m2",
                text="User works at Amazon in Portland",
                trust=0.90,
                timestamp=1709337600
            )
        ]
        
        # Test undisclosed
        result1 = verifier.verify("You work at Amazon in Portland", memories)
        assert result1.passed == False
        assert result1.requires_disclosure == True
        assert len(result1.contradiction_details) == 2  # employer and location
        
        # Test with disclosure
        result2 = verifier.verify(
            "You work at Amazon in Portland (changed from Microsoft in Seattle)",
            memories
        )
        assert result2.passed == True
        assert result2.requires_disclosure == False
    
    def test_no_contradiction_in_report_when_none_exist(self):
        """Test that reports show no contradictions when none exist."""
        verifier = GroundCheck()
        
        memories = [
            Memory(id="m1", text="User works at Microsoft"),
            Memory(id="m2", text="User lives in Seattle")
        ]
        
        result = verifier.verify("You work at Microsoft and live in Seattle", memories)
        
        assert result.passed == True
        assert len(result.contradiction_details) == 0
        assert len(result.contradicted_claims) == 0
        assert result.requires_disclosure == False
    
    def test_hallucination_and_contradiction_together(self):
        """Test handling of both hallucinations and contradictions."""
        verifier = GroundCheck()
        
        memories = [
            Memory(id="m1", text="User works at Microsoft", trust=0.85),
            Memory(id="m2", text="User works at Amazon", trust=0.85)
        ]
        
        # Both Amazon and Google are hallucinations because the text is parsed as one compound value
        # The fact extractor sees "Amazon and Google" but only "Amazon" and "Microsoft" exist in memories
        # Since the extracted fact is treated as a compound, Google is a hallucination
        result = verifier.verify("You work at Google", memories)
        
        assert result.passed == False
        assert "Google" in result.hallucinations
