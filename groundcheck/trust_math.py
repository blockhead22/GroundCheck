"""
Trust Math Engine — CRT (Contradiction-aware Reasoning & Trust) Core Mathematics.

Implements:
- Trust vs Confidence separation
- Drift detection and measurement
- Belief-weighted retrieval scoring
- SSE mode selection (Lossless/Cogni/Hybrid)
- Trust evolution equations (aligned/reinforced/contradicted)
- 6-rule contradiction detection
- Reflection triggers via volatility

Zero required dependencies — numpy is optional with pure-Python fallback.

Philosophy:
- Memory first (coherence over time)
- Honesty over performance (contradictions are signals, not bugs)
- Belief evolves slower than speech
"""

from typing import Callable, Dict, List, Tuple, Optional, Any, Sequence, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from difflib import SequenceMatcher
import math
import re
import logging

# Optional numpy import with pure-Python fallback
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    _NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Type alias for vectors — numpy array if available, else plain list of floats
Vector = Any  # Union[np.ndarray, List[float]]

# Transient state cues (mood/energy/temporary condition).
_TRANSIENT_STATE_WORDS = {
    "tired", "exhausted", "fatigued", "sleepy", "burned out",
    "sad", "down", "depressed", "depression", "anxious", "anxiety",
    "stressed", "overwhelmed", "okay", "ok", "fine", "good", "bad",
    "sick", "ill", "hurt", "hurting", "in pain", "recovering",
    "lonely", "upset", "angry", "frustrated", "confused",
}

_MOOD_SLOTS = {
    "mood", "feeling", "emotion", "emotions", "status",
    "user.mood", "user.feeling", "user.emotion",
}


@dataclass
class RuleScore:
    """Score from a single detection rule."""
    rule: str
    fired: bool
    confidence: float  # 0.0–1.0
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"rule": self.rule, "fired": self.fired, "confidence": self.confidence, "reason": self.reason}


@dataclass
class DetectionResult:
    """Rich result from contradiction detection with per-rule confidence scores.

    Attributes:
        is_contradiction: Whether a contradiction was detected (any rule above threshold).
        reason: Human-readable explanation for the top-firing rule.
        confidence: Aggregate confidence (max of all fired rule confidences).
        rule_scores: Per-rule scores for every rule evaluated.
        fired_rules: Convenience list of rules that fired.
    """
    is_contradiction: bool
    reason: str
    confidence: float
    rule_scores: List[RuleScore] = field(default_factory=list)

    @property
    def fired_rules(self) -> List[RuleScore]:
        return [r for r in self.rule_scores if r.fired]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_contradiction": self.is_contradiction,
            "reason": self.reason,
            "confidence": self.confidence,
            "rule_scores": [r.to_dict() for r in self.rule_scores],
        }


def _is_transient_state_value(value: Optional[str]) -> bool:
    if value is None:
        return False
    low = str(value).lower().strip()
    if not low:
        return False
    for word in _TRANSIENT_STATE_WORDS:
        needle = str(word).lower().strip()
        if not needle:
            continue
        if " " in needle:
            if re.search(rf"(^|\W){re.escape(needle)}($|\W)", low):
                return True
        else:
            if re.search(rf"\b{re.escape(needle)}\b", low):
                return True
    return False


# ============================================================================
# Pure-Python vector math fallback (used when numpy is not installed)
# ============================================================================

def _dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product of two sequences."""
    return sum(x * y for x, y in zip(a, b))


def _norm(a: Sequence[float]) -> float:
    """L2 norm of a sequence."""
    return math.sqrt(sum(x * x for x in a))


def _clip(value: float, lo: float, hi: float) -> float:
    """Clip value to [lo, hi]."""
    return max(lo, min(hi, value))


# ============================================================================
# Enums
# ============================================================================

class SSEMode(Enum):
    """SSE compression modes based on significance."""
    LOSSLESS = "L"   # Identity-critical, contradiction-heavy
    COGNI = "C"      # Fast sketch, "what it felt like"
    HYBRID = "H"     # Adaptive mix


class MemorySource(Enum):
    """Source of memory item."""
    USER = "user"
    SYSTEM = "system"
    FALLBACK = "fallback"
    EXTERNAL = "external"
    REFLECTION = "reflection"
    SELF_REFLECTION = "self_reflection"
    LLM_OUTPUT = "llm_output"


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CRTConfig:
    """CRT system configuration parameters.

    All thresholds are calibrated from stress-test analysis.
    Override via ``CRTConfig(**overrides)`` or ``load_from_calibration(path)``.
    """

    # Trust evolution rates
    eta_pos: float = 0.1          # Trust increase rate for aligned memories
    eta_reinforce: float = 0.05   # Reinforcement rate for validated memories
    eta_neg: float = 0.15         # Trust decrease rate for contradictions

    # Thresholds
    theta_align: float = 0.15     # Drift threshold for alignment
    theta_contra: float = 0.28    # Drift threshold for contradiction
    theta_min: float = 0.30       # Minimum drift for confidence-based contradiction
    theta_drop: float = 0.30      # Confidence drop threshold
    theta_fallback: float = 0.42  # Drift threshold for fallback contradictions

    # Reconstruction gates
    theta_intent: float = 0.5     # Intent alignment gate
    theta_mem: float = 0.38       # Memory alignment gate

    # Reflection triggers
    theta_reflect: float = 0.5    # Volatility threshold for reflection

    # Retrieval
    lambda_time: float = 86400.0  # Time constant (1 day in seconds)
    alpha_trust: float = 0.7      # Trust weight in retrieval (vs confidence)

    # Trust bounds
    tau_base: float = 0.7         # Base trust for new memories
    tau_fallback_cap: float = 0.3 # Max trust for fallback speech
    tau_train_min: float = 0.6    # Min trust for weight updates

    # SSE mode selection
    T_L: float = 0.7              # Lossless threshold
    T_C: float = 0.3              # Cogni threshold

    # SSE significance weights
    w_emotion: float = 0.2
    w_novelty: float = 0.25
    w_user_mark: float = 0.3
    w_contradiction: float = 0.15
    w_future: float = 0.1

    # Volatility weights
    beta_drift: float = 0.3
    beta_alignment: float = 0.25
    beta_contradiction: float = 0.3
    beta_fallback: float = 0.15

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dict."""
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "CRTConfig":
        """Create config from dict, ignoring unknown keys."""
        known = {f.name for f in CRTConfig.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return CRTConfig(**filtered)

    @staticmethod
    def load_from_calibration(
        calibration_path: str = "calibrated_thresholds.json",
    ) -> "CRTConfig":
        """Load CRTConfig with calibrated thresholds from file.

        Falls back to default values if calibration file is missing or invalid.

        Args:
            calibration_path: Path to calibrated thresholds JSON.
        """
        import json
        from pathlib import Path

        config = CRTConfig()

        try:
            threshold_file = Path(calibration_path)
            if not threshold_file.exists():
                logger.info(
                    "[CRT_CONFIG] Calibration file not found: %s, using defaults",
                    calibration_path,
                )
                return config

            with open(threshold_file) as f:
                data = json.load(f)

            if "green_zone" in data:
                config.theta_align = 1.0 - data["green_zone"]
                logger.info(
                    "[CRT_CONFIG] Loaded calibrated theta_align: %.3f",
                    config.theta_align,
                )

            if "red_zone" in data:
                config.theta_contra = 1.0 - data["red_zone"]
                logger.info(
                    "[CRT_CONFIG] Loaded calibrated theta_contra: %.3f",
                    config.theta_contra,
                )

            if "yellow_zone" in data:
                config.theta_fallback = 1.0 - data["yellow_zone"]
                logger.info(
                    "[CRT_CONFIG] Loaded calibrated theta_fallback: %.3f",
                    config.theta_fallback,
                )

            logger.info(
                "[CRT_CONFIG] Loaded calibrated thresholds from %s",
                calibration_path,
            )

        except Exception as e:
            logger.warning(
                "[CRT_CONFIG] Failed to load calibration from %s: %s, using defaults",
                calibration_path,
                e,
            )

        return config


# ============================================================================
# CRT Math Engine
# ============================================================================

class CRTMath:
    """Core CRT mathematical operations.

    Works with numpy arrays when available, falls back to plain lists otherwise.

    Implements:
    1. Similarity and drift measurement
    2. Trust-weighted retrieval scoring
    3. SSE mode selection
    4. Trust evolution (aligned / reinforced / contradicted)
    5. Reconstruction constraints
    6. 6-rule contradiction detection
    7. Reflection triggers via volatility
    """

    def __init__(self, config: Optional[CRTConfig] = None):
        self.config = config or CRTConfig()

    # ========================================================================
    # 1. Similarity and Drift
    # ========================================================================

    def similarity(self, a: Vector, b: Vector) -> float:
        """Cosine similarity between two vectors.

        ``sim(a, b) = (a . b) / (||a|| ||b||)``

        Works with numpy arrays or plain lists/tuples.
        """
        if a is None or b is None:
            return 0.0

        # Convert to list-like if needed
        a_seq: Sequence[float] = a if not _NUMPY_AVAILABLE else a
        b_seq: Sequence[float] = b if not _NUMPY_AVAILABLE else b

        if len(a_seq) == 0 or len(b_seq) == 0:
            return 0.0
        if len(a_seq) != len(b_seq):
            return 0.0

        if _NUMPY_AVAILABLE:
            a_arr = np.asarray(a, dtype=float)
            b_arr = np.asarray(b, dtype=float)
            norm_a = np.linalg.norm(a_arr)
            norm_b = np.linalg.norm(b_arr)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))
        else:
            norm_a = _norm(a_seq)
            norm_b = _norm(b_seq)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return _dot_product(a_seq, b_seq) / (norm_a * norm_b)

    def novelty(self, z_new: Vector, memory_vectors: List[Vector]) -> float:
        """Novelty of new input relative to stored memory.

        ``novelty(x) = 1 - max_i sim(z_new, z_i)``
        """
        if not memory_vectors:
            return 1.0
        max_sim = max(self.similarity(z_new, z_mem) for z_mem in memory_vectors)
        return 1.0 - max_sim

    def drift_meaning(self, z_new: Vector, z_prior: Vector) -> float:
        """Meaning drift between new output and prior belief.

        ``D_mean = 1 - sim(z_new, z_prior)``
        """
        return 1.0 - self.similarity(z_new, z_prior)

    # ========================================================================
    # 2. Trust-Weighted Retrieval Scoring
    # ========================================================================

    def recency_weight(self, t_memory: float, t_now: float) -> float:
        """Recency weighting with exponential decay.

        ``rho_i = exp(-(t_now - t_i) / lambda)``
        """
        delta_t = t_now - t_memory
        return math.exp(-delta_t / self.config.lambda_time)

    def belief_weight(self, trust: float, confidence: float) -> float:
        """Combined belief weight (trust + confidence).

        ``w_i = alpha * tau_i + (1 - alpha) * c_i``
        """
        alpha = self.config.alpha_trust
        return alpha * trust + (1 - alpha) * confidence

    def retrieval_score(
        self,
        similarity: float,
        recency: float,
        belief: float,
    ) -> float:
        """Final retrieval score.

        ``R_i = s_i * rho_i * w_i``
        """
        return similarity * recency * belief

    def compute_retrieval_scores(
        self,
        query_vector: Vector,
        memories: List[Dict[str, Any]],
        t_now: float,
    ) -> List[Tuple[int, float]]:
        """Compute retrieval scores for all memories.

        Returns list of ``(index, score)`` tuples sorted by score descending.
        Each memory dict should have keys: ``vector``, ``timestamp``, ``trust``, ``confidence``.
        """
        scores = []
        for i, mem in enumerate(memories):
            s_i = self.similarity(query_vector, mem["vector"])
            rho_i = self.recency_weight(mem["timestamp"], t_now)
            w_i = self.belief_weight(mem["trust"], mem["confidence"])
            R_i = self.retrieval_score(s_i, rho_i, w_i)
            scores.append((i, R_i))
        return sorted(scores, key=lambda x: x[1], reverse=True)

    # ========================================================================
    # 3. SSE Mode Selection
    # ========================================================================

    def compute_significance(
        self,
        emotion_intensity: float,
        novelty: float,
        user_marked: float,
        contradiction_signal: float,
        future_relevance: float,
    ) -> float:
        """Compute significance score for SSE mode selection.

        ``S = w1*e + w2*n + w3*u + w4*k + w5*f``
        """
        cfg = self.config
        return (
            cfg.w_emotion * emotion_intensity
            + cfg.w_novelty * novelty
            + cfg.w_user_mark * user_marked
            + cfg.w_contradiction * contradiction_signal
            + cfg.w_future * future_relevance
        )

    def select_sse_mode(self, significance: float) -> SSEMode:
        """Select SSE compression mode based on significance."""
        if significance >= self.config.T_L:
            return SSEMode.LOSSLESS
        elif significance <= self.config.T_C:
            return SSEMode.COGNI
        else:
            return SSEMode.HYBRID

    # ========================================================================
    # 4. Trust Evolution
    # ========================================================================

    def evolve_trust_aligned(self, tau_current: float, drift: float) -> float:
        """Trust evolution for aligned memories (low drift).

        ``tau_new = clip(tau + eta_pos * (1 - drift), 0, 1)``
        """
        tau_new = tau_current + self.config.eta_pos * (1.0 - drift)
        return _clip(tau_new, 0.0, 1.0)

    def evolve_trust_reinforced(self, tau_current: float, drift: float) -> float:
        """Trust reinforcement for validated memories.

        ``tau_new = clip(tau + eta_reinforce * (1 - drift), 0, 1)``
        """
        tau_new = tau_current + self.config.eta_reinforce * (1.0 - drift)
        return _clip(tau_new, 0.0, 1.0)

    def evolve_trust_contradicted(self, tau_current: float, drift: float) -> float:
        """Trust degradation for contradicted memories.

        ``tau_new = clip(tau * (1 - eta_neg * drift), 0, 1)``
        """
        tau_new = tau_current * (1.0 - self.config.eta_neg * drift)
        return _clip(tau_new, 0.0, 1.0)

    def cap_fallback_trust(self, tau: float, source: MemorySource) -> float:
        """Cap trust for fallback sources."""
        if source in {MemorySource.FALLBACK, MemorySource.LLM_OUTPUT}:
            return min(tau, self.config.tau_fallback_cap)
        return tau

    # ========================================================================
    # 5. Reconstruction Constraints (Holden Gates)
    # ========================================================================

    def intent_alignment(self, input_intent: Vector, output_intent: Vector) -> float:
        """Intent alignment score: ``sim(I(x), I(y))``."""
        return self.similarity(input_intent, output_intent)

    def memory_alignment(
        self,
        output_vector: Vector,
        retrieved_memories: List[Dict[str, Any]],
        retrieval_scores: List[float],
        output_text: str = "",
    ) -> float:
        """Memory alignment score (weighted by retrieval strength).

        ``A_mem = sum_i (softmax(R_i) * sim(E(y), z_i))``
        """
        if not retrieved_memories:
            return 0.0

        # Short fact extraction boost
        if output_text and len(output_text) < 50:
            output_lower = output_text.lower().strip()
            for mem in retrieved_memories[:3]:
                mem_text = mem.get("text", "").lower() if isinstance(mem.get("text"), str) else ""
                if output_lower and mem_text and output_lower in mem_text:
                    return 0.95

        # Softmax over retrieval scores
        if _NUMPY_AVAILABLE:
            scores_array = np.array(retrieval_scores)
            exp_scores = np.exp(scores_array - np.max(scores_array))
            weights = exp_scores / np.sum(exp_scores)
        else:
            max_s = max(retrieval_scores) if retrieval_scores else 0.0
            exp_scores = [math.exp(s - max_s) for s in retrieval_scores]
            sum_exp = sum(exp_scores)
            weights = [e / sum_exp for e in exp_scores] if sum_exp > 0 else [0.0] * len(exp_scores)

        alignment = 0.0
        for i, mem in enumerate(retrieved_memories):
            sim = self.similarity(output_vector, mem["vector"])
            w = weights[i] if _NUMPY_AVAILABLE else weights[i]
            alignment += float(w) * sim

        return alignment

    def check_reconstruction_gates(
        self,
        intent_align: float,
        memory_align: float,
        has_grounding_issues: bool = False,
        has_contradiction_issues: bool = False,
        has_extraction_issues: bool = False,
    ) -> Tuple[bool, str]:
        """Check if reconstruction passes gates (legacy v1).

        Returns ``(passed, reason)``.
        """
        if has_grounding_issues:
            return False, "grounding_fail"
        if has_contradiction_issues:
            return False, "contradiction_fail"
        if has_extraction_issues:
            return False, "extraction_fail"
        if intent_align < self.config.theta_intent:
            return False, f"intent_fail (align={intent_align:.3f} < {self.config.theta_intent})"
        if memory_align < self.config.theta_mem:
            return False, f"memory_fail (align={memory_align:.3f} < {self.config.theta_mem})"
        return True, "gates_passed"

    def check_reconstruction_gates_v2(
        self,
        intent_align: float,
        memory_align: float,
        response_type: str,
        grounding_score: float = 1.0,
        contradiction_severity: str = "none",
        blindspot_gate_boost: float = 0.0,
    ) -> Tuple[bool, str]:
        """Gradient gates with response-type awareness (v2).

        Response types: ``factual``, ``explanatory``, ``conversational``.
        Returns ``(passed, reason)``.
        """
        boost = max(0.0, min(blindspot_gate_boost, 0.15))
        boost_note = f" +boost={boost:.2f}" if boost > 0 else ""

        if contradiction_severity == "blocking":
            return False, "contradiction_fail"

        if response_type == "factual":
            t_intent = 0.35 + boost
            t_memory = 0.35 + boost
            t_ground = 0.30 + boost
            if intent_align < t_intent:
                return False, f"factual_intent_fail (align={intent_align:.3f} < {t_intent:.2f}{boost_note})"
            if memory_align < t_memory:
                return False, f"factual_memory_fail (align={memory_align:.3f} < {t_memory:.2f}{boost_note})"
            if grounding_score < t_ground:
                return False, f"factual_grounding_fail (score={grounding_score:.3f} < {t_ground:.2f}{boost_note})"

        elif response_type == "explanatory":
            t_intent = 0.35 + boost
            t_memory = 0.18 + boost
            t_ground = 0.20 + boost
            if intent_align < t_intent:
                return False, f"explanatory_intent_fail (align={intent_align:.3f} < {t_intent:.2f}{boost_note})"
            if memory_align < t_memory:
                return False, f"explanatory_memory_fail (align={memory_align:.3f} < {t_memory:.2f}{boost_note})"
            if grounding_score < t_ground:
                return False, f"explanatory_grounding_fail (score={grounding_score:.3f} < {t_ground:.2f}{boost_note})"

        else:  # conversational
            t_intent = 0.3 + boost
            if intent_align < t_intent:
                return False, f"conversational_intent_fail (align={intent_align:.3f} < {t_intent:.2f}{boost_note})"

        if contradiction_severity == "note":
            return True, "gates_passed_with_contradiction_note"

        return True, "gates_passed"

    # ========================================================================
    # 6. Contradiction Detection (6-rule system)
    # ========================================================================

    def detect_contradiction(
        self,
        drift: float,
        confidence_new: float,
        confidence_prior: float,
        source: MemorySource,
        text_new: str = "",
        text_prior: str = "",
        slot: Optional[str] = None,
        value_new: Optional[str] = None,
        value_prior: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Detect if a contradiction event should be triggered.

        6-rule system:
        0a. Entity swap detection
        0b. Negation contradiction
        0c. Preference/boolean inversion
        1.  Paraphrase tolerance (reduces false positives)
        2.  High drift > theta_contra
        3.  Confidence drop + moderate drift
        4.  Fallback source + drift

        Returns ``(is_contradiction, reason)``.
        """
        cfg = self.config

        # Rule 0a: Entity swap
        entity_swap, entity_reason = self._detect_entity_swap(
            slot, value_new, value_prior, text_new, text_prior,
        )
        if entity_swap:
            return True, entity_reason

        # Rule 0b: Negation
        negation_detected, negation_reason = self._detect_negation_contradiction(text_new, text_prior)
        if negation_detected:
            return True, negation_reason

        # Rule 0c: Preference/boolean inversion
        inversion_detected, inversion_reason = self._is_boolean_inversion(text_new, text_prior)
        if inversion_detected:
            return True, inversion_reason

        # Rule 1: Paraphrase tolerance
        if text_new and text_prior and drift > 0.35:
            if self._is_likely_paraphrase(text_new, text_prior, drift):
                return False, f"Paraphrase detected (drift={drift:.3f}, not contradiction)"

        # Rule 2: High drift
        if drift > cfg.theta_contra:
            return True, f"High drift: {drift:.3f} > {cfg.theta_contra}"

        # Rule 3: Confidence drop with moderate drift
        delta_c = confidence_prior - confidence_new
        if delta_c > cfg.theta_drop and drift > cfg.theta_min:
            return True, f"Confidence drop: Δc={delta_c:.3f}, drift={drift:.3f}"

        # Rule 4: Fallback source with drift
        if source in {MemorySource.FALLBACK, MemorySource.LLM_OUTPUT} and drift > cfg.theta_fallback:
            return True, f"Fallback drift: {drift:.3f} > {cfg.theta_fallback}"

        return False, "No contradiction"

    def detect_contradiction_scored(
        self,
        drift: float,
        confidence_new: float,
        confidence_prior: float,
        source: MemorySource,
        text_new: str = "",
        text_prior: str = "",
        slot: Optional[str] = None,
        value_new: Optional[str] = None,
        value_prior: Optional[str] = None,
        threshold: float = 0.5,
    ) -> DetectionResult:
        """Detect contradictions with per-rule confidence scores.

        Same 6-rule system as ``detect_contradiction`` but returns a
        ``DetectionResult`` with individual confidence scores per rule.
        Callers can threshold at any level; the default (0.5) matches the
        boolean behavior of ``detect_contradiction``.

        Args:
            drift: Semantic drift between old and new.
            confidence_new: Confidence in the new memory.
            confidence_prior: Confidence in the prior memory.
            source: Origin of the new memory.
            text_new: Full text of the new memory.
            text_prior: Full text of the prior memory.
            slot: Fact slot name (e.g. "employer").
            value_new: Extracted value from new text.
            value_prior: Extracted value from prior text.
            threshold: Minimum confidence to treat a rule as "fired". Default 0.5.

        Returns:
            ``DetectionResult`` with per-rule scores and aggregate confidence.
        """
        cfg = self.config
        scores: List[RuleScore] = []

        # Rule 0a: Entity swap
        entity_swap, entity_reason = self._detect_entity_swap(
            slot, value_new, value_prior, text_new, text_prior,
        )
        entity_conf = 0.95 if entity_swap else 0.0
        scores.append(RuleScore(rule="0a_entity_swap", fired=entity_swap, confidence=entity_conf, reason=entity_reason or "no entity swap"))

        # Rule 0b: Negation
        negation_detected, negation_reason = self._detect_negation_contradiction(text_new, text_prior)
        negation_conf = 0.90 if negation_detected else 0.0
        scores.append(RuleScore(rule="0b_negation", fired=negation_detected, confidence=negation_conf, reason=negation_reason or "no negation"))

        # Rule 0c: Preference/boolean inversion
        inversion_detected, inversion_reason = self._is_boolean_inversion(text_new, text_prior)
        inversion_conf = 0.85 if inversion_detected else 0.0
        scores.append(RuleScore(rule="0c_preference_inversion", fired=inversion_detected, confidence=inversion_conf, reason=inversion_reason or "no inversion"))

        # Rule 1: Paraphrase tolerance (suppressor — reduces confidence)
        is_paraphrase = False
        if text_new and text_prior and drift > 0.35:
            is_paraphrase = self._is_likely_paraphrase(text_new, text_prior, drift)
        scores.append(RuleScore(
            rule="1_paraphrase_tolerance", fired=is_paraphrase, confidence=0.80 if is_paraphrase else 0.0,
            reason=f"Paraphrase detected (drift={drift:.3f})" if is_paraphrase else "not a paraphrase",
        ))

        # Rule 2: High drift
        if drift > cfg.theta_contra:
            drift_conf = min(1.0, 0.5 + (drift - cfg.theta_contra) / (1.0 - cfg.theta_contra + 1e-9))
            drift_reason = f"High drift: {drift:.3f} > {cfg.theta_contra}"
            scores.append(RuleScore(rule="2_high_drift", fired=True, confidence=drift_conf, reason=drift_reason))
        else:
            scores.append(RuleScore(rule="2_high_drift", fired=False, confidence=0.0, reason=f"drift {drift:.3f} <= {cfg.theta_contra}"))

        # Rule 3: Confidence drop + moderate drift
        delta_c = confidence_prior - confidence_new
        rule3_fired = delta_c > cfg.theta_drop and drift > cfg.theta_min
        if rule3_fired:
            rule3_conf = min(1.0, 0.5 + delta_c)
            rule3_reason = f"Confidence drop: Δc={delta_c:.3f}, drift={drift:.3f}"
        else:
            rule3_conf = 0.0
            rule3_reason = f"No confidence drop (Δc={delta_c:.3f}, drift={drift:.3f})"
        scores.append(RuleScore(rule="3_confidence_drop", fired=rule3_fired, confidence=rule3_conf, reason=rule3_reason))

        # Rule 4: Fallback source + drift
        rule4_fired = source in {MemorySource.FALLBACK, MemorySource.LLM_OUTPUT} and drift > cfg.theta_fallback
        if rule4_fired:
            rule4_conf = min(1.0, 0.5 + (drift - cfg.theta_fallback))
            rule4_reason = f"Fallback drift: {drift:.3f} > {cfg.theta_fallback}"
        else:
            rule4_conf = 0.0
            rule4_reason = "No fallback drift"
        scores.append(RuleScore(rule="4_fallback_drift", fired=rule4_fired, confidence=rule4_conf, reason=rule4_reason))

        # Aggregate: paraphrase suppresses if no structural rules fired
        structural_fired = any(s.fired for s in scores if s.rule.startswith("0"))
        if is_paraphrase and not structural_fired:
            # Suppress drift-only rules
            for s in scores:
                if s.rule in ("2_high_drift", "3_confidence_drop", "4_fallback_drift") and s.fired:
                    s.fired = False
                    s.confidence *= 0.3  # Heavily reduce, don't zero

        fired = [s for s in scores if s.fired and s.confidence >= threshold]
        if fired:
            top = max(fired, key=lambda s: s.confidence)
            return DetectionResult(
                is_contradiction=True,
                reason=top.reason,
                confidence=top.confidence,
                rule_scores=scores,
            )

        return DetectionResult(
            is_contradiction=False,
            reason="No contradiction",
            confidence=0.0,
            rule_scores=scores,
        )

    def _is_likely_paraphrase(self, text_new: str, text_prior: str, drift: float) -> bool:
        """Check if two texts are paraphrases despite semantic drift."""
        if drift < 0.35 or drift > 0.50:
            return False

        def extract_key_elements(text: str) -> set:
            numbers = set(re.findall(r"\d+", text))
            caps = set(re.findall(r"(?<!^)(?<!\. )[A-Z][a-z]+", text))
            return numbers | caps

        keys_new = extract_key_elements(text_new)
        keys_prior = extract_key_elements(text_prior)

        # Explicit numeric mismatch
        nums_new = set(re.findall(r"\d+", text_new))
        nums_prior = set(re.findall(r"\d+", text_prior))
        if nums_new and nums_prior and nums_new != nums_prior:
            return False

        if keys_new and keys_prior:
            overlap = len(keys_new & keys_prior) / max(len(keys_new | keys_prior), 1)
            if overlap > 0.7:
                return True

        return False

    def _detect_entity_swap(
        self,
        slot: Optional[str],
        value_new: Optional[str],
        value_prior: Optional[str],
        text_new: str = "",
        text_prior: str = "",
    ) -> Tuple[bool, str]:
        """Detect entity swap: same slot but conflicting proper-noun values."""
        if not slot or value_new is None or value_prior is None:
            return False, ""

        candidate_new = str(value_new).strip()
        candidate_prior = str(value_prior).strip()
        if not candidate_new or not candidate_prior:
            return False, ""

        candidate_new_norm = candidate_new.lower()
        candidate_prior_norm = candidate_prior.lower()

        if candidate_new_norm == candidate_prior_norm:
            return False, ""

        if not (self._looks_like_entity(candidate_new) and self._looks_like_entity(candidate_prior)):
            return False, ""

        similarity = SequenceMatcher(None, candidate_new_norm, candidate_prior_norm).ratio()
        if similarity > 0.78:
            return False, ""

        return True, f"Entity swap detected for slot '{slot}': '{candidate_prior}' -> '{candidate_new}'"

    def _looks_like_entity(self, value: str) -> bool:
        """Heuristic: is value a named entity (proper noun or acronym)?"""
        if not value:
            return False
        tokens = re.findall(r"[A-Za-z][\w.&'-]*", value)
        return any(token[0].isupper() or token.isupper() for token in tokens)

    def _is_boolean_inversion(self, text_new: str, text_prior: str) -> Tuple[bool, str]:
        """Detect preference/boolean inversions (like vs dislike, prefer X vs prefer Y)."""
        if not text_new or not text_prior:
            return False, ""

        def extract_preferences(text: str) -> List[Tuple[str, str]]:
            patterns = [
                (r"\bprefer[s]?\s+(?P<obj>[^.;,!?:\n]+)", "prefer"),
                (r"\blike[s]?\s+(?P<obj>[^.;,!?:\n]+)", "like"),
                (r"\blove[s]?\s+(?P<obj>[^.;,!?:\n]+)", "like"),
                (r"\benjoy[s]?\s+(?P<obj>[^.;,!?:\n]+)", "like"),
                (r"\bdislike[s]?\s+(?P<obj>[^.;,!?:\n]+)", "dislike"),
                (r"\bhate[s]?\s+(?P<obj>[^.;,!?:\n]+)", "dislike"),
                (r"\bavoid[s]?\s+(?P<obj>[^.;,!?:\n]+)", "dislike"),
            ]
            preferences: List[Tuple[str, str]] = []
            lowered = text.lower()
            for pattern, label in patterns:
                for match in re.finditer(pattern, lowered):
                    obj = match.group("obj").strip()
                    obj = re.split(r"\b(but|however|though|although)\b", obj)[0].strip()
                    preferences.append((label, obj))
            return preferences

        def normalize_obj(obj: str) -> str:
            cleaned = re.sub(r"[^a-z0-9 ]+", " ", obj.lower())
            stopwords = {
                "a", "an", "the", "to", "in", "of", "on", "for",
                "with", "at", "my", "your", "our", "their", "his", "her",
            }
            return " ".join(word for word in cleaned.split() if word and word not in stopwords)

        def polarity(label: str) -> int:
            return 1 if label in {"prefer", "like"} else -1

        def objects_match(a: str, b: str) -> bool:
            if not a or not b:
                return False
            if a == b:
                return True
            ratio = SequenceMatcher(None, a, b).ratio()
            return ratio >= 0.75 or a in b or b in a

        prefs_new = extract_preferences(text_new)
        prefs_prior = extract_preferences(text_prior)

        if not prefs_new or not prefs_prior:
            return False, ""

        for label_new, obj_new_raw in prefs_new:
            obj_new = normalize_obj(obj_new_raw)
            if not obj_new:
                continue
            pol_new = polarity(label_new)

            for label_prior, obj_prior_raw in prefs_prior:
                obj_prior = normalize_obj(obj_prior_raw)
                if not obj_prior:
                    continue
                pol_prior = polarity(label_prior)

                same_target = objects_match(obj_new, obj_prior)
                if same_target and pol_new != pol_prior:
                    return True, f"Preference inversion on '{obj_new}'"

                if pol_new == pol_prior == 1 and not same_target:
                    return True, "Preference target changed"

        return False, ""

    def _detect_negation_contradiction(self, text_new: str, text_prior: str) -> Tuple[bool, str]:
        """Detect negation-based contradictions ('I don't X' vs 'I X')."""
        if not text_new or not text_prior:
            return False, ""

        text_new_lower = text_new.lower()
        text_prior_lower = text_prior.lower()

        negation_patterns = [
            (r"(?:i\s+)?(?:don'?t|do\s+not|no\s+longer|not\s+anymore)\s+(\w+(?:\s+\w+){0,3})", "negated"),
            (r"(?:i\s+)?(?:stopped|quit|left|no\s+longer)\s+(\w+(?:\s+\w+){0,3})", "ceased"),
            (r"(?:i'm\s+not|i\s+am\s+not)\s+(\w+(?:\s+\w+){0,3})", "negated_state"),
        ]

        negated_items = []
        for pattern, neg_type in negation_patterns:
            for match in re.finditer(pattern, text_new_lower):
                negated_items.append((match.group(1).strip(), neg_type))

        if not negated_items:
            return False, ""

        for item, neg_type in negated_items:
            item_words = item.split()[:3]
            item_pattern = r"\b" + r"\s+".join(re.escape(w) for w in item_words) + r"\b"

            if re.search(item_pattern, text_prior_lower):
                prior_negated = any(
                    re.search(p[0], text_prior_lower) for p in negation_patterns
                )
                if not prior_negated:
                    return True, f"Negation contradiction: '{item}' negated in new, affirmed in prior"

        return False, ""

    # ========================================================================
    # 6b. Context-Aware Contradiction Detection
    # ========================================================================

    def is_true_contradiction_contextual(
        self,
        slot: Optional[str],
        value_new: Optional[str],
        value_prior: Optional[str],
        temporal_status_new: str = "active",
        temporal_status_prior: str = "active",
        domains_new: Optional[list] = None,
        domains_prior: Optional[list] = None,
        drift: float = 0.0,
    ) -> Tuple[bool, str]:
        """Context-aware contradiction detection.

        Reduces false positives by considering temporal status, domain context,
        and transient state.  Only flags when same slot, overlapping time,
        overlapping domains, and mutually exclusive values.

        Returns ``(is_contradiction, reason)``.
        """
        if not slot:
            return False, "no_slot_no_contradiction"

        if value_new is None or value_prior is None:
            return False, "missing_values"

        value_new_norm = str(value_new).lower().strip()
        value_prior_norm = str(value_prior).lower().strip()
        slot_lower = str(slot).lower()

        # Transient state updates are not contradictions
        if slot_lower in _MOOD_SLOTS:
            return False, "transient_state_slot_update"
        if _is_transient_state_value(value_new_norm) or _is_transient_state_value(value_prior_norm):
            return False, "transient_state_update"

        if value_new_norm == value_prior_norm:
            return False, "same_value"

        # Handle "LEFT:" prefix for employer negations
        if value_new_norm.startswith("left:"):
            left_value = value_new_norm.replace("left:", "").strip()
            if left_value == value_prior_norm:
                return False, "temporal_update_left_employer"

        if temporal_status_new == "past" and temporal_status_prior == "past":
            return False, "both_past_no_conflict"

        if temporal_status_new == "past" and temporal_status_prior == "active":
            return False, "temporal_deprecation"

        domains_new = domains_new or ["general"]
        domains_prior = domains_prior or ["general"]
        domains_new_set = set(domains_new)
        domains_prior_set = set(domains_prior)

        has_general = "general" in domains_new_set or "general" in domains_prior_set
        has_specific_overlap = bool(domains_new_set & domains_prior_set - {"general"})

        if not has_general and not has_specific_overlap:
            return False, "different_domains_coexist"

        if temporal_status_new in ("future", "potential") and temporal_status_prior in ("future", "potential"):
            return False, "future_plans_no_conflict"

        return True, f"true_contradiction: same slot '{slot}', overlapping context, different values"

    def _is_numeric_contradiction(
        self, value_new: str, value_prior: str, threshold: float = 0.20,
    ) -> Tuple[bool, str]:
        """Check if two numeric values contradict (>20% difference by default)."""
        try:
            match_new = re.search(r"[\d.]+", str(value_new))
            match_prior = re.search(r"[\d.]+", str(value_prior))

            if not match_new or not match_prior:
                return False, "not_numeric"

            num_new = float(match_new.group())
            num_prior = float(match_prior.group())

            if num_prior == 0:
                is_contra = num_new != 0
                return is_contra, "numeric_zero_comparison" if is_contra else "both_zero"

            diff_pct = abs(num_new - num_prior) / abs(num_prior)
            if diff_pct > threshold:
                return True, f"numeric_drift_{diff_pct:.0%}"
            return False, f"numeric_within_tolerance_{diff_pct:.0%}"

        except (AttributeError, ValueError, TypeError) as e:
            return False, f"not_numeric: {e}"

    def classify_fact_change(
        self,
        slot: str,
        value_new: str,
        value_prior: str,
        text_new: str = "",
        text_prior: str = "",
    ) -> str:
        """Classify the type of fact change.

        Returns one of: ``refinement``, ``revision``, ``temporal``, ``conflict``.
        """
        text_new_lower = (text_new or "").lower()
        value_new_lower = str(value_new).lower()
        value_prior_lower = str(value_prior).lower()

        revision_markers = [
            "actually", "correction", "i meant", "i mean", "to clarify",
            "wrong", "mistake", "not", "no longer", "left", "quit",
        ]
        if any(marker in text_new_lower for marker in revision_markers):
            return "revision"

        temporal_markers = [
            "now", "currently", "recently", "promoted", "moved to",
            "started", "new", "changed to",
        ]
        if any(marker in text_new_lower for marker in temporal_markers):
            return "temporal"

        if value_prior_lower in value_new_lower:
            return "refinement"

        geographic_refinement_pairs = [
            ("seattle", "bellevue"), ("new york", "brooklyn"),
            ("los angeles", "santa monica"), ("san francisco", "oakland"),
        ]
        for general, specific in geographic_refinement_pairs:
            if value_prior_lower == general and specific in value_new_lower:
                return "refinement"

        return "conflict"

    # ========================================================================
    # 7. Reflection Triggers
    # ========================================================================

    def compute_volatility(
        self,
        drift: float,
        memory_alignment: float,
        is_contradiction: bool,
        is_fallback: bool,
    ) -> float:
        """Compute volatility/instability score.

        ``V = beta1*D_mean + beta2*(1 - A_mem) + beta3*contradiction + beta4*fallback``
        """
        cfg = self.config
        contra_flag = 1.0 if is_contradiction else 0.0
        fallback_flag = 1.0 if is_fallback else 0.0
        return (
            cfg.beta_drift * drift
            + cfg.beta_alignment * (1.0 - memory_alignment)
            + cfg.beta_contradiction * contra_flag
            + cfg.beta_fallback * fallback_flag
        )

    def should_reflect(self, volatility: float) -> bool:
        """Check if reflection should be triggered (``V >= theta_reflect``)."""
        return volatility >= self.config.theta_reflect

    # ========================================================================
    # 8. Safety Boundaries
    # ========================================================================

    def can_train_on_memory(
        self,
        trust: float,
        has_open_contradiction: bool,
        source: MemorySource,
    ) -> Tuple[bool, str]:
        """Check if memory can be used for training/weight updates."""
        if trust < self.config.tau_train_min:
            return False, f"Trust too low: {trust:.3f} < {self.config.tau_train_min}"
        if has_open_contradiction:
            return False, "Open contradiction exists"
        if source in {MemorySource.FALLBACK, MemorySource.LLM_OUTPUT}:
            return False, "Fallback source not verified"
        return True, "Safe to train"


# ============================================================================
# Utility Functions
# ============================================================================

def encode_vector(text: str, encoder=None) -> Any:
    """Encode text to semantic vector.

    Uses provided encoder callback, falls back to hash-based vector.
    """
    if encoder is not None:
        return encoder(text)

    # Hash-based fallback — deterministic pseudo-embedding
    import hashlib

    hash_obj = hashlib.sha256(text.encode())
    hash_bytes = hash_obj.digest()

    # Convert each byte to a float in [-1, 1] range
    raw = [(float(b) - 127.5) / 127.5 for b in hash_bytes[:32]]

    if _NUMPY_AVAILABLE:
        vector = np.array(raw, dtype=np.float64)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector
    else:
        n = _norm(raw)
        if n > 0:
            raw = [x / n for x in raw]
        return raw


def extract_emotion_intensity(text: str) -> float:
    """Extract emotion intensity from text (0-1). Simple heuristic."""
    emotion_words = ["love", "hate", "fear", "angry", "happy", "sad", "excited", "worried"]

    intensity = 0.0

    # Exclamation marks
    intensity += min(text.count("!") * 0.1, 0.3)

    # Caps ratio
    if len(text) > 0:
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
        intensity += min(caps_ratio * 0.5, 0.3)

    # Emotion words
    text_lower = text.lower()
    emotion_count = sum(1 for word in emotion_words if word in text_lower)
    intensity += min(emotion_count * 0.1, 0.4)

    return min(intensity, 1.0)
