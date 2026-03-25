"""Tests for groundcheck.ml_detector — ML contradiction detection."""

import pytest
from groundcheck.ml_detector import (
    _has_retraction_pattern,
    _extract_remainder_after_retraction,
    _is_semantic_equivalent,
    _is_detail_enrichment,
    _is_transient_state_value,
    _names_are_related,
    MLContradictionDetector,
)


class TestRetractionPatterns:
    def test_detects_actually_no(self):
        assert _has_retraction_pattern("actually no, i do work there")

    def test_detects_wait_no(self):
        assert _has_retraction_pattern("wait no, that's wrong")

    def test_no_false_positives(self):
        assert not _has_retraction_pattern("i think so")
        assert not _has_retraction_pattern("yes that's right")

    def test_remainder_extraction(self):
        result = _extract_remainder_after_retraction("actually no, i do have a phd")
        assert "i do have a phd" in result


class TestSemanticEquivalence:
    def test_phd_doctorate(self):
        assert _is_semantic_equivalent("PhD", "doctorate")

    def test_ml_machine_learning(self):
        assert _is_semantic_equivalent("ML", "machine learning")

    def test_developer_engineer(self):
        assert _is_semantic_equivalent("developer", "engineer")

    def test_different_companies_not_equivalent(self):
        assert not _is_semantic_equivalent("Google", "Microsoft")

    def test_exact_match(self):
        assert _is_semantic_equivalent("hello", "hello")

    def test_substring_enrichment(self):
        assert _is_semantic_equivalent("dog", "rescue dog")


class TestDetailEnrichment:
    def test_enrichment_detected(self):
        assert _is_detail_enrichment("dog", "rescue dog")
        assert _is_detail_enrichment("Max", "Max the golden retriever")

    def test_not_enrichment(self):
        assert not _is_detail_enrichment("Google", "Microsoft")


class TestTransientState:
    def test_mood_words_detected(self):
        assert _is_transient_state_value("tired")
        assert _is_transient_state_value("feeling anxious today")

    def test_non_transient(self):
        assert not _is_transient_state_value("software engineer")
        assert not _is_transient_state_value("PhD in computer science")


class TestNameRelatedness:
    def test_substring_names(self):
        assert _names_are_related("Max", "Max the dog")

    def test_exact_match(self):
        assert _names_are_related("Alice", "Alice")

    def test_different_names(self):
        assert not _names_are_related("Alice", "Bob")


class TestMLDetectorFallback:
    """Test the detector with heuristic fallback (no models loaded)."""

    def test_fallback_detects_contradiction(self):
        detector = MLContradictionDetector(model_dir=None)
        result = detector.check_contradiction(
            old_value="Google",
            new_value="Microsoft",
            slot="employer",
        )
        assert result["is_contradiction"] is True

    def test_fallback_transient_not_contradiction(self):
        detector = MLContradictionDetector(model_dir=None)
        result = detector.check_contradiction(
            old_value="creative thinker",
            new_value="feeling tired",
            slot="mood",
        )
        assert result["is_contradiction"] is False
        assert result["category"] == "TEMPORAL"

    def test_fallback_retraction_forces_conflict(self):
        detector = MLContradictionDetector(model_dir=None)
        result = detector.check_contradiction(
            old_value="no PhD",
            new_value="PhD",
            slot="education",
            context={"query": "actually no, I do have a PhD"},
        )
        assert result["is_contradiction"] is True

    def test_fallback_same_value_no_contradiction(self):
        detector = MLContradictionDetector(model_dir=None)
        result = detector.check_contradiction(
            old_value="Google",
            new_value="Google",
            slot="employer",
        )
        assert result["is_contradiction"] is False
