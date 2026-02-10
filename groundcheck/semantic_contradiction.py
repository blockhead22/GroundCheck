"""Semantic contradiction detection using NLI models."""

from typing import List, Optional, Tuple
from dataclasses import dataclass

_nli_pipeline = None

@dataclass 
class ContradictionResult:
    """Result of contradiction check."""
    is_contradiction: bool
    confidence: float
    method: str  # "slot_based", "nli", "hybrid"
    explanation: str

class SemanticContradictionDetector:
    """
    Detect contradictions using semantic understanding.
    
    Uses Natural Language Inference (NLI) as fallback for
    cases where slot-based detection is insufficient.
    """
    
    def __init__(
        self,
        use_nli: bool = True,
        nli_model: str = "cross-encoder/nli-deberta-v3-small",
        contradiction_threshold: float = 0.7
    ):
        self.use_nli = use_nli
        self.nli_model_name = nli_model
        self.contradiction_threshold = contradiction_threshold
        self._nli = None
    
    def _get_nli_pipeline(self):
        """Lazy load NLI model."""
        global _nli_pipeline
        if _nli_pipeline is None and self.use_nli:
            try:
                from transformers import pipeline
                _nli_pipeline = pipeline(
                    "zero-shot-classification",
                    model=self.nli_model_name
                )
            except ImportError:
                self.use_nli = False
            except Exception as e:
                print(f"Warning: Could not load NLI model: {e}")
                self.use_nli = False
        return _nli_pipeline
    
    def check_contradiction(
        self,
        statement_a: str,
        statement_b: str,
        slot: Optional[str] = None
    ) -> ContradictionResult:
        """
        Check if two statements contradict each other.
        
        Args:
            statement_a: First statement
            statement_b: Second statement  
            slot: Optional fact slot for context
            
        Returns:
            ContradictionResult with determination and confidence
        """
        # Quick check: identical statements don't contradict
        if statement_a.lower().strip() == statement_b.lower().strip():
            return ContradictionResult(
                is_contradiction=False,
                confidence=1.0,
                method="exact_match",
                explanation="Statements are identical"
            )
        
        # Try NLI if available
        if self.use_nli:
            nli = self._get_nli_pipeline()
            if nli:
                try:
                    # Check if A contradicts B
                    result = nli(
                        statement_a,
                        candidate_labels=["contradiction", "entailment", "neutral"],
                        hypothesis_template="This statement is {} with: " + statement_b
                    )
                    
                    contradiction_score = 0.0
                    for label, score in zip(result["labels"], result["scores"]):
                        if label == "contradiction":
                            contradiction_score = score
                            break
                    
                    is_contradiction = contradiction_score >= self.contradiction_threshold
                    
                    return ContradictionResult(
                        is_contradiction=is_contradiction,
                        confidence=contradiction_score,
                        method="nli",
                        explanation=f"NLI contradiction score: {contradiction_score:.2f}"
                    )
                except Exception as e:
                    print(f"NLI check failed: {e}")
        
        # Fallback: simple heuristic
        # If same slot with different values, likely contradiction
        return ContradictionResult(
            is_contradiction=True,  # Conservative: assume contradiction
            confidence=0.5,
            method="heuristic",
            explanation="Different values for same fact type"
        )
