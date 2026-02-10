"""
Tuple-Based Ground Check Verifier for CRT System.

Extends the GroundCheck system to support tuple-based fact comparison.
This allows verification of LLM-generated answers against both hard facts
(regex-extracted) and open-world tuples (LLM-extracted).

Design:
- Compares answer tuples against memory tuples
- Uses semantic similarity for value matching
- Supports both exact and fuzzy matching
- Integrates with existing GroundCheck system
"""

from __future__ import annotations

import logging
from typing import List, Optional, Set, Tuple, Dict, Any

from .types import Memory, VerificationReport
from .verifier import GroundCheck

logger = logging.getLogger(__name__)

# Tuple-based verification requires the full CRT project (personal_agent).
# In the standalone PyPI package, this is always disabled.
try:
    from personal_agent.fact_tuples import FactTuple, FactTupleSet, FactAction
    from personal_agent.llm_extractor import LLMFactExtractor
    HAS_TUPLES = True
except ImportError:
    HAS_TUPLES = False


class TupleGroundCheck(GroundCheck):
    """
    Extended GroundCheck that supports tuple-based fact verification.
    
    This class extends the base GroundCheck to compare extracted fact tuples
    from generated answers against fact tuples stored in memory. It provides:
    
    1. Tuple extraction from answer text
    2. Semantic comparison of tuple values
    3. Support for different match levels (exact, fuzzy, semantic)
    4. Hallucination detection at the tuple level
    
    Example:
        >>> verifier = TupleGroundCheck()
        >>> memory_tuples = [FactTuple(entity="User", attribute="employer", value="Microsoft")]
        >>> result = verifier.verify_answer_tuples(
        ...     "You work at Amazon.",
        ...     memory_tuples
        ... )
        >>> print(result["is_supported"])  # False
        >>> print(result["hallucinations"])  # [FactTuple(employer=Amazon)]
    """
    
    # Semantic similarity threshold for value matching
    TUPLE_SIMILARITY_THRESHOLD = 0.85
    
    # Attributes that require exact match (no fuzzy matching)
    EXACT_MATCH_ATTRIBUTES = {
        "name",
        "age",
        "graduation_year",
        "account_number",
        "phone_number",
        "email",
    }
    
    def __init__(self):
        """Initialize TupleGroundCheck with LLM extractor."""
        super().__init__()
        
        # Initialize LLM extractor for answer parsing
        if HAS_TUPLES:
            try:
                self.llm_extractor = LLMFactExtractor()
            except Exception as e:
                logger.warning(f"Failed to initialize LLM extractor: {e}")
                self.llm_extractor = None
        else:
            self.llm_extractor = None
    
    def verify_answer_tuples(
        self,
        answer: str,
        memory_tuples: List['FactTuple'],
        strict: bool = False,
    ) -> Dict[str, Any]:
        """
        Verify that answer claims are supported by memory tuples.
        
        Args:
            answer: Generated answer text to verify
            memory_tuples: List of fact tuples from memory
            strict: If True, require exact matches only
            
        Returns:
            Dictionary with:
                - is_supported: bool, whether all claims are supported
                - supported_claims: List of tuples that found support
                - hallucinations: List of tuples not found in memory
                - match_details: Details about how claims were matched
        """
        if not HAS_TUPLES:
            return {
                "is_supported": True,  # Can't verify without tuples
                "supported_claims": [],
                "hallucinations": [],
                "match_details": [],
                "error": "Tuple verification unavailable",
            }
        
        if not answer or not memory_tuples:
            return {
                "is_supported": True,
                "supported_claims": [],
                "hallucinations": [],
                "match_details": [],
            }
        
        # Extract claims from answer
        answer_tuples = self._extract_answer_tuples(answer)
        
        supported = []
        hallucinations = []
        match_details = []
        
        for claimed in answer_tuples:
            is_match, matched_memory, match_type = self._find_support(
                claimed, memory_tuples, strict
            )
            
            if is_match:
                supported.append(claimed)
                match_details.append({
                    "claimed": claimed.to_dict(),
                    "matched": matched_memory.to_dict() if matched_memory else None,
                    "match_type": match_type,
                })
            else:
                hallucinations.append(claimed)
                match_details.append({
                    "claimed": claimed.to_dict(),
                    "matched": None,
                    "match_type": "no_match",
                })
        
        return {
            "is_supported": len(hallucinations) == 0,
            "supported_claims": supported,
            "hallucinations": hallucinations,
            "match_details": match_details,
        }
    
    def _extract_answer_tuples(self, answer: str) -> List['FactTuple']:
        """
        Extract fact tuples from answer text.
        
        Uses LLM extractor if available, otherwise returns empty list.
        
        Args:
            answer: Answer text to extract from
            
        Returns:
            List of FactTuple objects
        """
        if self.llm_extractor is None:
            return []
        
        try:
            return self.llm_extractor.extract_tuples(answer)
        except Exception as e:
            logger.warning(f"Failed to extract tuples from answer: {e}")
            return []
    
    def _find_support(
        self,
        claimed: 'FactTuple',
        memory_tuples: List['FactTuple'],
        strict: bool = False,
    ) -> Tuple[bool, Optional['FactTuple'], str]:
        """
        Find memory support for a claimed tuple.
        
        Args:
            claimed: The tuple being claimed in the answer
            memory_tuples: List of tuples from memory
            strict: If True, require exact matches
            
        Returns:
            Tuple of (is_supported, matching_memory_tuple, match_type)
        """
        for mem_tuple in memory_tuples:
            # Check entity match (allow "User" to match "I", "me", etc.)
            if not self._entities_match(claimed.entity, mem_tuple.entity):
                continue
            
            # Check attribute match (fuzzy for related attributes)
            if not self._attributes_match(claimed.attribute, mem_tuple.attribute):
                continue
            
            # Check value match
            is_match, match_type = self._values_match(
                claimed.value,
                mem_tuple.value,
                claimed.attribute,
                strict,
            )
            
            if is_match:
                return True, mem_tuple, match_type
        
        return False, None, "no_match"
    
    def _entities_match(self, claimed_entity: str, memory_entity: str) -> bool:
        """
        Check if two entities refer to the same thing.
        
        Handles common aliases like "User" == "I" == "me" == "myself".
        """
        claimed_lower = claimed_entity.lower().strip()
        memory_lower = memory_entity.lower().strip()
        
        # Direct match
        if claimed_lower == memory_lower:
            return True
        
        # User entity aliases
        user_aliases = {"user", "i", "me", "myself"}
        if claimed_lower in user_aliases and memory_lower in user_aliases:
            return True
        
        return False
    
    def _attributes_match(
        self,
        claimed_attr: str,
        memory_attr: str,
    ) -> bool:
        """
        Check if two attributes refer to the same property.
        
        Handles related attributes like "employer" and "company".
        """
        claimed_lower = claimed_attr.lower().strip().replace(" ", "_")
        memory_lower = memory_attr.lower().strip().replace(" ", "_")
        
        # Direct match
        if claimed_lower == memory_lower:
            return True
        
        # Related attribute groups
        related_groups = [
            {"employer", "company", "workplace", "employment_company", "work_at"},
            {"location", "city", "residence", "lives_in", "home"},
            {"name", "full_name", "identity_name"},
            {"title", "job_title", "role", "position"},
            {"age", "years_old"},
            {"hobby", "hobbies", "pastime", "interest"},
        ]
        
        for group in related_groups:
            if claimed_lower in group and memory_lower in group:
                return True
        
        return False
    
    def _values_match(
        self,
        claimed_value: str,
        memory_value: str,
        attribute: str,
        strict: bool = False,
    ) -> Tuple[bool, str]:
        """
        Check if two values match (exact, fuzzy, or semantic).
        
        Args:
            claimed_value: Value from the answer
            memory_value: Value from memory
            attribute: Attribute name (for context-aware matching)
            strict: If True, only allow exact matches
            
        Returns:
            Tuple of (is_match, match_type)
        """
        claimed_norm = claimed_value.lower().strip()
        memory_norm = memory_value.lower().strip()
        
        # Exact match
        if claimed_norm == memory_norm:
            return True, "exact"
        
        if strict:
            return False, "no_match"
        
        # Attributes requiring exact match
        if attribute.lower() in self.EXACT_MATCH_ATTRIBUTES:
            return False, "no_match"
        
        # Substring match (one contains the other)
        if claimed_norm in memory_norm or memory_norm in claimed_norm:
            return True, "substring"
        
        # Fuzzy string matching
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, claimed_norm, memory_norm).ratio()
        if ratio >= 0.85:
            return True, "fuzzy"
        
        # Semantic similarity (if embeddings available)
        if self.embedding_model is not None:
            try:
                emb_claimed = self.embedding_model.encode(claimed_value)
                emb_memory = self.embedding_model.encode(memory_value)
                
                # Cosine similarity
                import numpy as np
                similarity = float(np.dot(emb_claimed, emb_memory) / (
                    np.linalg.norm(emb_claimed) * np.linalg.norm(emb_memory)
                ))
                
                if similarity >= self.TUPLE_SIMILARITY_THRESHOLD:
                    return True, "semantic"
                    
            except Exception as e:
                logger.debug(f"Semantic matching failed: {e}")
        
        return False, "no_match"
    
    def verify_with_tuples(
        self,
        text: str,
        memories: List[Memory],
        memory_tuples: Optional[List['FactTuple']] = None,
    ) -> VerificationReport:
        """
        Combined verification using both traditional and tuple-based methods.
        
        This method runs the standard GroundCheck verification and optionally
        enhances it with tuple-based verification.
        
        Args:
            text: Generated text to verify
            memories: List of Memory objects
            memory_tuples: Optional list of fact tuples for additional verification
            
        Returns:
            VerificationReport with combined results
        """
        # Run standard verification
        report = self.verify(text, memories)
        
        # Enhance with tuple verification if tuples provided
        if memory_tuples and HAS_TUPLES:
            tuple_result = self.verify_answer_tuples(text, memory_tuples)
            
            # Add tuple hallucinations to report
            if not tuple_result["is_supported"]:
                for hall in tuple_result["hallucinations"]:
                    hall_value = f"{hall.attribute}={hall.value}"
                    if hall_value not in report.hallucinations:
                        report.hallucinations.append(hall_value)
                
                # Update passed status
                report.passed = report.passed and tuple_result["is_supported"]
        
        return report


def create_tuple_verifier() -> TupleGroundCheck:
    """
    Factory function to create a TupleGroundCheck instance.
    
    Returns:
        Configured TupleGroundCheck instance
    """
    return TupleGroundCheck()
