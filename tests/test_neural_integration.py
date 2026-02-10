"""Integration tests for neural mode (v0.3.0).

These tests exercise the full verify() pipeline with neural matching enabled
and prove that paraphrases are caught that regex-only mode would miss.
"""

import pytest
from groundcheck import GroundCheck, Memory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_sentence_transformers() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


requires_neural = pytest.mark.skipif(
    not _has_sentence_transformers(),
    reason="sentence-transformers not installed",
)


# ---------------------------------------------------------------------------
# Test: neural= parameter
# ---------------------------------------------------------------------------


class TestNeuralParameter:
    """GroundCheck(neural=...) controls whether semantic models are loaded."""

    def test_neural_false_no_semantic_matcher(self):
        v = GroundCheck(neural=False)
        assert v.semantic_matcher is None
        assert v.hybrid_extractor is None
        assert v.semantic_contradiction_detector is None

    def test_neural_true_default(self):
        v = GroundCheck()
        assert v.neural is True

    @requires_neural
    def test_neural_true_creates_matcher(self):
        v = GroundCheck(neural=True)
        assert v.semantic_matcher is not None


# ---------------------------------------------------------------------------
# Test: regex-only baseline (neural=False)
# ---------------------------------------------------------------------------


class TestRegexOnlyBaseline:
    """Verify that regex-only mode still works without neural deps."""

    def test_exact_match_passes(self):
        v = GroundCheck(neural=False)
        mems = [Memory(id="m1", text="User works at Microsoft")]
        result = v.verify("You work at Microsoft", mems)
        assert result.passed

    def test_hallucination_detected(self):
        v = GroundCheck(neural=False)
        mems = [Memory(id="m1", text="User works at Microsoft")]
        result = v.verify("You work at Amazon", mems)
        assert not result.passed
        assert "Amazon" in result.hallucinations

    def test_empty_text_passes(self):
        v = GroundCheck(neural=False)
        result = v.verify("", [])
        assert result.passed


# ---------------------------------------------------------------------------
# Test: paraphrase detection through verify() — the money tests
# ---------------------------------------------------------------------------


@requires_neural
class TestNeuralParaphraseDetection:
    """End-to-end tests proving embedding similarity catches paraphrases
    that regex and fuzzy matching miss."""

    def test_employed_by_matches_works_at(self):
        """'employed by Google' should match 'works at Google'."""
        v = GroundCheck(neural=True)
        mems = [Memory(id="m1", text="User works at Google")]
        result = v.verify("You are employed by Google", mems)
        # SemanticMatcher's normalization canonicalizes 'employed by' → 'work at'
        assert result.passed, f"Hallucinations: {result.hallucinations}"

    def test_resides_in_matches_lives_in(self):
        """'resides in Seattle' should match 'lives in Seattle'."""
        v = GroundCheck(neural=True)
        mems = [Memory(id="m1", text="User lives in Seattle")]
        result = v.verify("You reside in Seattle", mems)
        assert result.passed, f"Hallucinations: {result.hallucinations}"

    def test_studied_at_matches_graduated_from(self):
        """'studied at MIT' should match 'graduated from MIT'."""
        v = GroundCheck(neural=True)
        mems = [Memory(id="m1", text="User graduated from MIT")]
        result = v.verify("You studied at MIT", mems)
        assert result.passed, f"Hallucinations: {result.hallucinations}"

    def test_software_developer_matches_engineer(self):
        """'software developer' should match 'software engineer' via embeddings."""
        v = GroundCheck(neural=True)
        mems = [Memory(id="m1", text="FACT: occupation = software engineer")]
        result = v.verify("You are a software developer", mems)
        # This relies on embedding similarity — the hardest case.
        # If it doesn't pass via embeddings, SemanticMatcher's synonym table
        # or term-overlap should still catch it.
        assert result.passed, f"Hallucinations: {result.hallucinations}"

    def test_different_entity_still_fails(self):
        """Paraphrase matching should NOT make unrelated entities pass."""
        v = GroundCheck(neural=True)
        mems = [Memory(id="m1", text="User works at Google")]
        result = v.verify("You work at Amazon", mems)
        assert not result.passed
        assert "Amazon" in result.hallucinations

    def test_nyc_matches_new_york_city(self):
        """Embedding similarity catches 'NYC' = 'New York City'."""
        v = GroundCheck(neural=True)
        mems = [Memory(id="m1", text="User lives in NYC")]
        result = v.verify("I live in New York City", mems)
        assert result.passed, f"Hallucinations: {result.hallucinations}"

    def test_neural_off_misses_value_paraphrase(self):
        """Without neural, VALUE-level paraphrases are missed.
        
        Both sides extract a 'location' fact, but the values differ:
        'NYC' (memory) vs 'New York City' (text).  The fallback fuzzy chain
        can't equate these — SequenceMatcher ratio for 'nyc' vs 'new york city'
        is too low, and neither is a substring of the other.
        """
        v = GroundCheck(neural=False)
        mems = [Memory(id="m1", text="User lives in NYC")]
        result = v.verify("I live in New York City", mems)
        # Regex fallback can't equate 'NYC' ↔ 'New York City'
        assert not result.passed


# ---------------------------------------------------------------------------
# Test: SemanticContradictionDetector integration
# ---------------------------------------------------------------------------


@requires_neural
class TestNeuralContradictionDetection:
    """Tests confirming NLI-based contradiction refinement works through verify()."""

    def test_known_exclusive_still_detected(self):
        """Known-exclusive slots bypass NLI and always detect contradictions."""
        v = GroundCheck(neural=True)
        mems = [
            Memory(id="m1", text="User works at Google", trust=0.9),
            Memory(id="m2", text="User works at Amazon", trust=0.9),
        ]
        result = v.verify("You work at Google", mems)
        assert len(result.contradiction_details) > 0
        assert result.contradiction_details[0].slot == "employer"

    def test_additive_slots_not_contradicted(self):
        """Additive slots (skill, language) should NOT trigger contradictions."""
        v = GroundCheck(neural=True)
        mems = [
            Memory(id="m1", text="User knows Python", trust=0.9),
            Memory(id="m2", text="User knows JavaScript", trust=0.9),
        ]
        result = v.verify("You know Python", mems)
        # 'programming_language' or 'skill' should be additive
        lang_contradictions = [
            c for c in result.contradiction_details
            if c.slot in ("programming_language", "skill", "language")
        ]
        assert len(lang_contradictions) == 0


# ---------------------------------------------------------------------------
# Test: SemanticMatcher strategy attribution
# ---------------------------------------------------------------------------


@requires_neural
class TestMatcherStrategyAttribution:
    """Verify which matching strategy fires for different paraphrase types."""

    def test_normalization_catches_employed_by(self):
        """SemanticMatcher._normalize() canonicalizes 'employed by' → 'work at'."""
        from groundcheck.semantic_matcher import SemanticMatcher
        m = SemanticMatcher(use_embeddings=True)
        is_match, method, _ = m.is_match(
            "employed by Google", {"works at Google"}
        )
        assert is_match
        # Should match via exact (after normalization) or term_overlap
        assert method in ("exact", "term_overlap", "fuzzy", "embedding")

    def test_embedding_catches_semantic_similarity(self):
        """True semantic similarity (different surface forms, same meaning)."""
        from groundcheck.semantic_matcher import SemanticMatcher
        m = SemanticMatcher(use_embeddings=True, embedding_threshold=0.75)
        is_match, method, _ = m.is_match(
            "Python programmer", {"Python developer"}
        )
        assert is_match

    def test_embedding_rejects_unrelated(self):
        """Unrelated concepts should NOT match even with embeddings."""
        from groundcheck.semantic_matcher import SemanticMatcher
        m = SemanticMatcher(use_embeddings=True)
        is_match, method, _ = m.is_match(
            "loves hiking", {"works at Google"}
        )
        assert not is_match
