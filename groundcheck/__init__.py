"""GroundCheck - Trust-weighted hallucination detection for AI agents.

Verify LLM-generated text against multiple sources with contradiction awareness.
Zero dependencies. Sub-2ms.

Example:
    >>> from groundcheck import GroundCheck, Memory
    >>> 
    >>> verifier = GroundCheck()
    >>> memories = [Memory(id="m1", text="User works at Microsoft")]
    >>> 
    >>> result = verifier.verify("You work at Amazon", memories)
    >>> print(result.passed)  # False
    >>> print(result.hallucinations)  # ["Amazon"]
"""

__version__ = "1.0.0"

from .types import Memory, VerificationReport, ExtractedFact, ContradictionDetail
from .verifier import GroundCheck
from .fact_extractor import extract_fact_slots
from .knowledge_extractor import (
    extract_knowledge_facts,
    extract_knowledge_facts_detailed,
    KnowledgeFact,
    infer_facts,
    find_entities,
    find_verbs,
)

# Neural extraction and semantic matching (optional)
try:
    from .neural_extractor import HybridFactExtractor, NeuralExtractionResult
    from .semantic_matcher import SemanticMatcher
    from .semantic_contradiction import SemanticContradictionDetector, ContradictionResult
    _NEURAL_AVAILABLE = True
except ImportError:
    _NEURAL_AVAILABLE = False
    HybridFactExtractor = None
    NeuralExtractionResult = None
    SemanticMatcher = None
    SemanticContradictionDetector = None
    ContradictionResult = None

__all__ = [
    "GroundCheck",
    "Memory",
    "VerificationReport",
    "ExtractedFact",
    "ContradictionDetail",
    "extract_fact_slots",
    "extract_knowledge_facts",
    "extract_knowledge_facts_detailed",
    "KnowledgeFact",
    "infer_facts",
    "find_entities",
    "find_verbs",
    "HybridFactExtractor",
    "NeuralExtractionResult",
    "SemanticMatcher",
    "SemanticContradictionDetector",
    "ContradictionResult",
]
