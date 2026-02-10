"""Semantic matching for paraphrase detection."""

from typing import List, Optional, Set, Tuple
from difflib import SequenceMatcher
import re

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None
    _HAS_NUMPY = False

# Global cache: Intentionally shared across instances to avoid reloading heavy models.
# This is standard practice in ML libraries to save memory and startup time.
# Thread safety: SentenceTransformer models are thread-safe for encoding.
_embedding_model = None

class SemanticMatcher:
    """
    Semantic matching with multiple fallback strategies.
    
    Strategies (in order):
    1. Exact match (fastest)
    2. Normalized match (remove articles, lowercase)
    3. Fuzzy match (SequenceMatcher)
    4. Synonym expansion (hardcoded synonyms)
    5. Embedding similarity (slowest, most accurate)
    """
    
    # Common paraphrase patterns
    SYNONYMS = {
        "employer": {
            "works at": ["employed by", "employed at", "job at", "works for", "working at", "working for"],
            "employee of": ["works at", "employed by"],
        },
        "location": {
            "lives in": ["resides in", "based in", "located in", "living in"],
            "from": ["originally from", "comes from", "hometown"],
        },
        "occupation": {
            "software engineer": ["swe", "software developer", "programmer", "coder", "dev"],
            "data scientist": ["ds", "ml engineer", "machine learning engineer"],
            "product manager": ["pm", "product lead"],
        }
    }
    
    def __init__(
        self,
        use_embeddings: bool = True,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_threshold: float = 0.85
    ):
        self.use_embeddings = use_embeddings
        self.embedding_model_name = embedding_model
        self.embedding_threshold = embedding_threshold
        self._model = None
    
    def _get_embedding_model(self):
        """Lazy load embedding model."""
        global _embedding_model
        if _embedding_model is None and self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                _embedding_model = SentenceTransformer(self.embedding_model_name)
            except ImportError:
                print("Warning: sentence-transformers not installed")
                self.use_embeddings = False
            except Exception as e:
                print(f"Warning: Could not load embedding model: {e}")
                self.use_embeddings = False
        return _embedding_model
    
    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        t = text.lower().strip()
        # Canonicalize common paraphrase forms into stable templates.
        t = re.sub(
            r'\b(employed by|employed at|works for|working for|working at|works at|job at)\b',
            'work at',
            t,
        )
        t = re.sub(
            r'\b(resides in|based in|located in|living in)\b',
            'live in',
            t,
        )
        t = re.sub(
            r'\b(graduated from|graduate from|studied at|study at|attended|went to)\b',
            'study at',
            t,
        )
        # Normalize educational suffix noise.
        t = re.sub(r'\buniversity\b', '', t)
        t = re.sub(r'\b(a|an|the)\b', '', t)
        t = re.sub(r'[^a-z0-9\s]', ' ', t)
        t = ' '.join(t.split())
        return t
    
    def _fuzzy_match(self, a: str, b: str, threshold: float = 0.85) -> bool:
        """Fuzzy string matching."""
        return SequenceMatcher(None, a, b).ratio() >= threshold
    
    def _synonym_match(self, claimed: str, supported: str, slot: str) -> bool:
        """Check if claimed and supported are synonyms."""
        claimed_norm = self._normalize(claimed)
        supported_norm = self._normalize(supported)
        
        slot_synonyms = self.SYNONYMS.get(slot, {})
        
        for base, variants in slot_synonyms.items():
            all_forms = [base] + variants
            all_forms_norm = [self._normalize(f) for f in all_forms]
            
            if claimed_norm in all_forms_norm and supported_norm in all_forms_norm:
                return True
        
        return False
    
    def _embedding_match(self, claimed: str, supported: str) -> bool:
        """Check semantic similarity via embeddings."""
        if not _HAS_NUMPY:
            return False
        model = self._get_embedding_model()
        if model is None:
            return False
        
        try:
            embeddings = model.encode([claimed, supported])
            # Use cosine similarity (normalized dot product)
            emb1, emb2 = embeddings[0], embeddings[1]
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            # Handle zero vectors
            if norm1 == 0 or norm2 == 0:
                return False
            
            similarity = float(np.dot(emb1, emb2) / (norm1 * norm2))
            return similarity >= self.embedding_threshold
        except Exception:
            return False
    
    def is_match(
        self,
        claimed: str,
        supported_values: Set[str],
        slot: str = ""
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Check if claimed value matches any supported value.
        
        Returns:
            (is_match, method_used, matched_value)
        """
        claimed_norm = self._normalize(claimed)
        
        for supported in supported_values:
            supported_norm = self._normalize(supported)

            # Strategy 0: Slot-aware synonym match should take precedence over
            # normalization-based exact matching so method attribution remains
            # meaningful in diagnostics/tests.
            if slot and self._synonym_match(claimed, supported, slot):
                return True, "synonym", supported
            
            # Strategy 1: Exact match
            if claimed_norm == supported_norm:
                return True, "exact", supported
            
            # Strategy 2: Fuzzy match
            if self._fuzzy_match(claimed_norm, supported_norm):
                return True, "fuzzy", supported
            
            # Strategy 3: Substring (for compound values)
            if claimed_norm in supported_norm or supported_norm in claimed_norm:
                return True, "substring", supported
            
            # Strategy 4b: Term-overlap for short factual phrases.
            claimed_terms = set(claimed_norm.split())
            supported_terms = set(supported_norm.split())
            if claimed_terms and supported_terms:
                overlap = len(claimed_terms & supported_terms) / len(claimed_terms)
                if overlap >= 0.67:
                    return True, "term_overlap", supported
        
        # Strategy 5: Embedding match (slowest, only if others fail)
        if self.use_embeddings:
            for supported in supported_values:
                if self._embedding_match(claimed, supported):
                    return True, "embedding", supported
        
        return False, "none", None
