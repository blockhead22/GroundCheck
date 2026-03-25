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

__version__ = "2.0.0"

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

# Trust math (core CRT engine)
from .trust_math import CRTConfig, CRTMath, SSEMode, MemorySource, DetectionResult, RuleScore

# Storage backends
from .backends import InMemoryBackend, SQLiteBackend, LedgerBackend

# Contradiction ledger
from .ledger import (
    ContradictionLedger,
    ContradictionEntry,
    ContradictionStatus,
    ContradictionType,
)

# Lifecycle engine
from .lifecycle import (
    ContradictionLifecycle,
    ContradictionLifecycleState,
    ContradictionLifecycleEntry,
    DisclosurePolicy,
    TransparencyLevel,
    MemoryStyle,
    UserTransparencyPrefs,
    LifecycleHook,
)

# Trust decay
from .decay import run_trust_decay_pass, reinforce_memory

# Trace logging
from .trace_logger import ContradictionTraceLogger, get_trace_logger

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

# ML detection (optional — requires sklearn)
try:
    from .ml_detector import MLContradictionDetector
    _ML_DETECTOR_AVAILABLE = True
except ImportError:
    _ML_DETECTOR_AVAILABLE = False
    MLContradictionDetector = None  # type: ignore

__all__ = [
    # Core verification (v1)
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
    # Neural (optional)
    "HybridFactExtractor",
    "NeuralExtractionResult",
    "SemanticMatcher",
    "SemanticContradictionDetector",
    "ContradictionResult",
    # Trust math (v2)
    "CRTConfig",
    "CRTMath",
    "SSEMode",
    "MemorySource",
    # Storage backends (v2.1)
    "InMemoryBackend",
    "SQLiteBackend",
    "LedgerBackend",
    # Scored detection (v2.1)
    "DetectionResult",
    "RuleScore",
    # Contradiction ledger (v2)
    "ContradictionLedger",
    "ContradictionEntry",
    "ContradictionStatus",
    "ContradictionType",
    # Lifecycle engine (v2)
    "ContradictionLifecycle",
    "ContradictionLifecycleState",
    "ContradictionLifecycleEntry",
    "DisclosurePolicy",
    "TransparencyLevel",
    "MemoryStyle",
    "UserTransparencyPrefs",
    "LifecycleHook",
    # Trust decay (v2)
    "run_trust_decay_pass",
    "reinforce_memory",
    # Trace logging (v2)
    "ContradictionTraceLogger",
    "get_trace_logger",
    # ML detection (v2, optional)
    "MLContradictionDetector",
]
