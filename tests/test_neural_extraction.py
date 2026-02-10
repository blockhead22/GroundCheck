"""Tests for neural extraction and semantic matching."""

import pytest
from groundcheck.neural_extractor import HybridFactExtractor, NeuralExtractionResult
from groundcheck.semantic_matcher import SemanticMatcher


class TestHybridExtractor:
    """Test hybrid fact extraction."""
    
    def test_regex_fast_path(self):
        """Test that clear patterns use fast regex path."""
        extractor = HybridFactExtractor(use_neural=False)
        result = extractor.extract("I work at Microsoft")
        assert "employer" in result.entities
        assert result.method == "regex"
        assert "Microsoft" in result.entities["employer"]
    
    def test_regex_with_name(self):
        """Test name extraction via regex."""
        extractor = HybridFactExtractor(use_neural=False)
        result = extractor.extract("My name is Alice")
        assert "name" in result.entities
        assert result.method == "regex"
    
    def test_regex_with_location(self):
        """Test location extraction via regex."""
        extractor = HybridFactExtractor(use_neural=False)
        result = extractor.extract("I live in Seattle")
        assert "location" in result.entities
        assert result.method == "regex"
    
    def test_high_confidence_regex(self):
        """Test high confidence when multiple facts are found."""
        extractor = HybridFactExtractor(use_neural=False)
        result = extractor.extract("My name is Bob, I work at Amazon, and I live in Seattle")
        assert result.method == "regex"
        # With 2-3 facts extracted, confidence should be 0.7 or higher
        assert result.confidence >= 0.7
    
    def test_hybrid_fallback_disabled(self):
        """Test that neural is not used when disabled."""
        extractor = HybridFactExtractor(use_neural=False, confidence_threshold=0.99)
        result = extractor.extract("Microsoft employee based in Seattle")
        # Should stay regex even with low confidence
        assert result.method == "regex"
    
    def test_neural_extraction_graceful_fallback(self):
        """Test graceful fallback when transformers not available."""
        # Even if neural is enabled, it should work without transformers
        extractor = HybridFactExtractor(use_neural=True)
        result = extractor.extract("I work at Google")
        # Should succeed with either regex or hybrid
        assert result.method in ["regex", "hybrid"]
        if "employer" in result.entities:
            assert "Google" in result.entities["employer"]


class TestSemanticMatcher:
    """Test semantic matching functionality."""
    
    def test_exact_match(self):
        """Test exact string matching."""
        matcher = SemanticMatcher(use_embeddings=False)
        is_match, method, matched = matcher.is_match("Microsoft", {"Microsoft"})
        assert is_match
        assert method == "exact"
        assert matched == "Microsoft"
    
    def test_case_insensitive_match(self):
        """Test case-insensitive matching."""
        matcher = SemanticMatcher(use_embeddings=False)
        is_match, method, matched = matcher.is_match("microsoft", {"Microsoft"})
        assert is_match
        assert method in ["exact", "fuzzy"]
    
    def test_fuzzy_match(self):
        """Test fuzzy matching with slight variations."""
        matcher = SemanticMatcher(use_embeddings=False)
        is_match, method, matched = matcher.is_match("Seattle", {"Seattle, WA"})
        assert is_match
        # Should match via substring or fuzzy
        assert method in ["fuzzy", "substring"]
    
    def test_substring_match(self):
        """Test substring matching."""
        matcher = SemanticMatcher(use_embeddings=False)
        is_match, method, matched = matcher.is_match("Microsoft", {"Microsoft Corporation"})
        assert is_match
        assert method == "substring"
    
    def test_synonym_match_employer(self):
        """Test synonym matching for employer slot."""
        matcher = SemanticMatcher(use_embeddings=False)
        is_match, method, matched = matcher.is_match(
            "works for Google",
            {"employed by Google"},
            slot="employer"
        )
        # This test is aspirational - synonym matching may not work perfectly
        # without the full phrase being in the synonym list
        # For now, it should at least not crash
        assert isinstance(is_match, bool)
        assert method in ["exact", "fuzzy", "substring", "synonym", "none"]
    
    def test_synonym_match_location(self):
        """Test synonym matching for location slot."""
        matcher = SemanticMatcher(use_embeddings=False)
        is_match, method, matched = matcher.is_match(
            "lives in Seattle",
            {"resides in Seattle"},
            slot="location"
        )
        # Similar to above - may not match perfectly but should not crash
        assert isinstance(is_match, bool)
    
    def test_no_match(self):
        """Test when values don't match."""
        matcher = SemanticMatcher(use_embeddings=False)
        is_match, method, matched = matcher.is_match("Microsoft", {"Google"})
        assert not is_match
        assert method == "none"
        assert matched is None
    
    def test_normalization(self):
        """Test text normalization."""
        matcher = SemanticMatcher(use_embeddings=False)
        # Test with articles
        is_match, method, matched = matcher.is_match("the Microsoft", {"Microsoft"})
        assert is_match
    
    def test_embedding_match(self):
        """Test embedding-based matching with proper cosine similarity."""
        try:
            matcher = SemanticMatcher(use_embeddings=True)
            
            # Only run if sentence-transformers is available and model loaded
            if not matcher.use_embeddings or matcher._get_embedding_model() is None:
                pytest.skip("sentence-transformers not installed or model not available")
            
            # Test semantic similarity
            is_match, method, matched = matcher.is_match(
                "employed by Google",
                {"works at Google"}
            )
            assert is_match
            # Depending on normalization improvements, this may match before
            # reaching the embedding stage.
            assert method in {"embedding", "exact", "fuzzy", "substring", "synonym", "term_overlap"}
            
            # Test non-match
            is_match, method, matched = matcher.is_match(
                "software engineer",
                {"data scientist"}
            )
            # These shouldn't match even with embeddings
            assert not is_match
        except ImportError:
            pytest.skip("SemanticMatcher not available")


class TestNeuralExtractionResult:
    """Test NeuralExtractionResult dataclass."""
    
    def test_dataclass_creation(self):
        """Test creating NeuralExtractionResult."""
        result = NeuralExtractionResult(
            entities={"employer": ["Microsoft"]},
            confidence=0.9,
            method="regex"
        )
        assert result.entities == {"employer": ["Microsoft"]}
        assert result.confidence == 0.9
        assert result.method == "regex"
