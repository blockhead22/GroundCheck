"""Type definitions for GroundCheck library."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Memory:
    """A memory or retrieved context item.
    
    Attributes:
        id: Unique identifier for the memory
        text: The text content of the memory
        trust: Trust score between 0.0 and 1.0 (default: 1.0)
        metadata: Optional additional metadata
        timestamp: Optional Unix timestamp (seconds since epoch)
    """
    id: str
    text: str
    trust: float = 1.0
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[int] = None


@dataclass
class ExtractedFact:
    """A fact extracted from text.
    
    Attributes:
        slot: The fact category/type (e.g., "name", "employer", "location")
        value: The actual value of the fact
        normalized: Normalized/lowercased version for matching
    """
    slot: str
    value: Any
    normalized: str


@dataclass
class ContradictionDetail:
    """Details about a contradiction between memories.
    
    Attributes:
        slot: Fact slot/category (e.g., "employer", "location")
        values: List of contradicting values (e.g., ["Microsoft", "Amazon"])
        memory_ids: List of memory IDs containing each value
        timestamps: List of Unix timestamps for each memory (None if unavailable)
        trust_scores: List of trust scores for each memory
    """
    slot: str
    values: List[str]
    memory_ids: List[str]
    timestamps: List[Optional[int]]
    trust_scores: List[float]
    
    @property
    def most_recent_value(self) -> str:
        """Return value from most recent memory.
        
        Uses timestamps if available, otherwise falls back to highest trust score.
        """
        if not self.timestamps or all(t is None for t in self.timestamps):
            # No timestamps - use highest trust (same as most_trusted_value)
            return self.most_trusted_value
        
        # Filter out None timestamps and pair with values
        valid_pairs = [(t, v) for t, v in zip(self.timestamps, self.values) if t is not None]
        if valid_pairs:
            return max(valid_pairs, key=lambda x: x[0])[1]
        
        # All timestamps are None, fall back to trust
        return self.most_trusted_value
    
    @property
    def most_trusted_value(self) -> str:
        """Return value from most trusted memory."""
        max_idx = self.trust_scores.index(max(self.trust_scores))
        return self.values[max_idx]


@dataclass
class VerificationReport:
    """Results of grounding verification.
    
    Attributes:
        original: The original generated text
        corrected: Corrected text (if mode="strict"), or None
        passed: Whether verification passed (no hallucinations or undisclosed contradictions)
        hallucinations: List of detected hallucinated values
        grounding_map: Mapping from claim to supporting memory ID
        confidence: Confidence score for the verification (0.0-1.0)
        facts_extracted: Facts extracted from the generated text
        facts_supported: Facts that were found in memories
        contradicted_claims: Claims that rely on contradicted facts
        contradiction_details: Full contradiction information
        requires_disclosure: True if output should acknowledge contradiction
        expected_disclosure: Suggested disclosure text
    """
    original: str
    corrected: Optional[str] = None
    passed: bool = True
    hallucinations: List[str] = field(default_factory=list)
    grounding_map: Dict[str, str] = field(default_factory=dict)
    confidence: float = 1.0
    facts_extracted: Dict[str, ExtractedFact] = field(default_factory=dict)
    facts_supported: Dict[str, ExtractedFact] = field(default_factory=dict)
    contradicted_claims: List[str] = field(default_factory=list)
    contradiction_details: List['ContradictionDetail'] = field(default_factory=list)
    requires_disclosure: bool = False
    expected_disclosure: Optional[str] = None
