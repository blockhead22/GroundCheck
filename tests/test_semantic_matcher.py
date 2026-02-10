"""Tests for semantic matcher functionality."""

import pytest
from groundcheck.semantic_matcher import SemanticMatcher


class TestSemanticMatcherNormalization:
    """Test text normalization in SemanticMatcher."""
    
    def test_normalize_basic(self):
        """Test basic text normalization."""
        matcher = SemanticMatcher(use_embeddings=False)
        normalized = matcher._normalize("Hello World")
        assert normalized == "hello world"
    
    def test_normalize_articles(self):
        """Test that articles are removed."""
        matcher = SemanticMatcher(use_embeddings=False)
        normalized = matcher._normalize("the Microsoft")
        assert normalized == "microsoft"
        
        normalized = matcher._normalize("a company")
        assert normalized == "company"
    
    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        matcher = SemanticMatcher(use_embeddings=False)
        normalized = matcher._normalize("multiple   spaces")
        assert normalized == "multiple spaces"
    
    def test_normalize_empty(self):
        """Test normalizing empty string."""
        matcher = SemanticMatcher(use_embeddings=False)
        normalized = matcher._normalize("")
        assert normalized == ""


class TestSemanticMatcherFuzzy:
    """Test fuzzy matching."""
    
    def test_fuzzy_exact(self):
        """Test fuzzy match with identical strings."""
        matcher = SemanticMatcher(use_embeddings=False)
        assert matcher._fuzzy_match("test", "test")
    
    def test_fuzzy_similar(self):
        """Test fuzzy match with similar strings."""
        matcher = SemanticMatcher(use_embeddings=False)
        assert matcher._fuzzy_match("hello", "hallo", threshold=0.8)
    
    def test_fuzzy_different(self):
        """Test fuzzy match with different strings."""
        matcher = SemanticMatcher(use_embeddings=False)
        assert not matcher._fuzzy_match("hello", "world")


class TestSemanticMatcherSynonyms:
    """Test synonym matching."""
    
    def test_synonym_match_employer_works_at(self):
        """Test employer synonyms with 'works at'."""
        matcher = SemanticMatcher(use_embeddings=False)
        result = matcher._synonym_match("works at", "employed by", "employer")
        assert result
    
    def test_synonym_match_employer_variations(self):
        """Test various employer synonym variations."""
        matcher = SemanticMatcher(use_embeddings=False)
        
        # Test pairs from the synonym list
        assert matcher._synonym_match("works at", "works for", "employer")
        assert matcher._synonym_match("employed by", "employed at", "employer")
    
    def test_synonym_match_location(self):
        """Test location synonyms."""
        matcher = SemanticMatcher(use_embeddings=False)
        assert matcher._synonym_match("lives in", "resides in", "location")
        assert matcher._synonym_match("lives in", "based in", "location")
    
    def test_synonym_match_occupation(self):
        """Test occupation synonyms."""
        matcher = SemanticMatcher(use_embeddings=False)
        assert matcher._synonym_match("software engineer", "software developer", "occupation")
        assert matcher._synonym_match("software engineer", "programmer", "occupation")
    
    def test_synonym_no_match_different_slots(self):
        """Test that synonyms don't match across different slots."""
        matcher = SemanticMatcher(use_embeddings=False)
        # These aren't in the synonym list
        result = matcher._synonym_match("random1", "random2", "employer")
        assert not result


class TestSemanticMatcherIntegration:
    """Test full is_match functionality."""
    
    def test_is_match_exact(self):
        """Test exact matching through is_match."""
        matcher = SemanticMatcher(use_embeddings=False)
        is_match, method, matched = matcher.is_match("test", {"test", "other"})
        assert is_match
        assert method == "exact"
        assert matched in ["test", "other"]
    
    def test_is_match_substring(self):
        """Test substring matching."""
        matcher = SemanticMatcher(use_embeddings=False)
        is_match, method, matched = matcher.is_match(
            "Microsoft",
            {"Microsoft Corporation", "Google"}
        )
        assert is_match
        assert method in ["exact", "substring"]
    
    def test_is_match_no_match(self):
        """Test when there's no match."""
        matcher = SemanticMatcher(use_embeddings=False)
        is_match, method, matched = matcher.is_match(
            "Apple",
            {"Microsoft", "Google"}
        )
        assert not is_match
        assert method == "none"
        assert matched is None
    
    def test_is_match_with_slot_context(self):
        """Test matching with slot context for synonyms."""
        matcher = SemanticMatcher(use_embeddings=False)
        is_match, method, matched = matcher.is_match(
            "works at",
            {"employed by"},
            slot="employer"
        )
        assert is_match
        assert method == "synonym"
    
    def test_is_match_multiple_candidates(self):
        """Test matching against multiple candidates."""
        matcher = SemanticMatcher(use_embeddings=False)
        is_match, method, matched = matcher.is_match(
            "Seattle",
            {"New York", "Seattle", "Boston"}
        )
        assert is_match
        assert matched == "Seattle"


class TestSemanticMatcherEmbeddings:
    """Test embedding-based matching (graceful degradation)."""
    
    def test_embedding_model_lazy_load(self):
        """Test that embedding model is lazy loaded."""
        matcher = SemanticMatcher(use_embeddings=True)
        # Model should not be loaded yet
        assert matcher._model is None
    
    def test_embedding_disabled_fallback(self):
        """Test that matcher works when embeddings are disabled."""
        matcher = SemanticMatcher(use_embeddings=False)
        is_match, method, matched = matcher.is_match(
            "Microsoft",
            {"Microsoft Corporation"}
        )
        # Should still work with non-embedding methods
        assert is_match
        assert method in ["exact", "fuzzy", "substring"]
    
    def test_embedding_graceful_failure(self):
        """Test graceful failure when embeddings unavailable."""
        matcher = SemanticMatcher(use_embeddings=True)
        # Try to use embeddings - should gracefully fall back
        is_match, method, matched = matcher.is_match(
            "test",
            {"test"}
        )
        # Should still match via exact match
        assert is_match
