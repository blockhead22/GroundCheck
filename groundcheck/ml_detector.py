"""
ML-Based Contradiction Detection.

Uses trained XGBoost models to detect contradictions and recommend resolution
policies. Falls back to heuristic detection when sklearn/xgboost unavailable.

Optional dependencies: ``scikit-learn``, ``xgboost``.
Install via: ``pip install groundcheck[ml]``
"""

import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, Any, Optional, Set

logger = logging.getLogger(__name__)

# Optional sklearn/numpy imports
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    _SKLEARN_AVAILABLE = False

# ============================================================================
# Retraction Pattern Detection
# ============================================================================

RETRACTION_PATTERNS = [
    "actually no,", "actually no ",
    "wait no,", "wait no ",
    "no wait,", "no wait ",
]


def _has_retraction_pattern(text: str) -> bool:
    return any(text.startswith(p) or f" {p}" in text for p in RETRACTION_PATTERNS)


def _extract_remainder_after_retraction(text: str) -> str:
    for pattern in RETRACTION_PATTERNS:
        if pattern in text:
            return text.split(pattern, 1)[-1]
    return text


def _is_meaningful_substring(substring: str, full_text: str) -> bool:
    if len(substring) > 5:
        return True
    return full_text.startswith(substring) or full_text.endswith(substring)


# ============================================================================
# Semantic Equivalence Database
# ============================================================================

SEMANTIC_EQUIVALENTS: Dict[str, Set[str]] = {
    "phd": {"doctorate", "doctoral", "ph.d.", "doctor of philosophy", "doctoral degree"},
    "doctorate": {"phd", "doctoral", "ph.d.", "doctor of philosophy"},
    "masters": {"master's", "ms", "ma", "msc", "master of science", "master of arts"},
    "bachelor": {"bachelor's", "bs", "ba", "bsc", "undergraduate"},
    "ml": {"machine learning", "ai", "artificial intelligence", "deep learning"},
    "ai": {"artificial intelligence", "ml", "machine learning", "deep learning"},
    "cs": {"computer science", "computing", "comp sci"},
    "data science": {"data analytics", "analytics", "data engineering"},
    "developer": {"engineer", "programmer", "coder", "software engineer"},
    "engineer": {"developer", "programmer", "software developer"},
    "scientist": {"researcher", "analyst"},
    "married": {"spouse", "husband", "wife", "partner"},
    "dog": {"pup", "puppy", "pet", "canine"},
    "cat": {"kitty", "kitten", "pet", "feline"},
}

DETAIL_ENRICHMENT_WORDS = {
    "rescue", "adopted", "beloved", "new", "old", "favorite",
    "senior", "junior", "lead", "chief", "principal", "staff",
    "golden", "black", "white", "brown",
    "downtown", "metro", "greater",
}

TRANSIENT_STATE_WORDS = {
    "tired", "exhausted", "fatigued", "sleepy", "burned out",
    "sad", "down", "depressed", "depression", "anxious", "anxiety",
    "stressed", "overwhelmed", "okay", "ok", "fine", "good", "bad",
    "sick", "ill", "hurt", "hurting", "in pain", "recovering",
    "lonely", "upset", "angry", "frustrated", "confused",
}


def _is_transient_state_value(value: str) -> bool:
    low = str(value).lower()
    return any(word in low for word in TRANSIENT_STATE_WORDS)


def _is_semantic_equivalent(old_value: str, new_value: str) -> bool:
    """Check if two values are semantically equivalent (not a contradiction)."""
    old_lower = str(old_value).lower().strip()
    new_lower = str(new_value).lower().strip()

    if old_lower == new_lower:
        return True

    # Substring / detail enrichment
    if old_lower in new_lower and _is_meaningful_substring(old_lower, new_lower):
        return True
    if new_lower in old_lower and _is_meaningful_substring(new_lower, old_lower):
        return True

    # Synonym database
    old_words = set(old_lower.split())
    new_words = set(new_lower.split())

    for key, synonyms in SEMANTIC_EQUIVALENTS.items():
        all_forms = {key} | synonyms
        old_has = bool(old_words & all_forms)
        new_has = bool(new_words & all_forms)
        if not (old_has and new_has):
            for form in all_forms:
                if " " in form:
                    if form in old_lower:
                        old_has = True
                    if form in new_lower:
                        new_has = True
        if old_has and new_has:
            return True

    return False


def _is_detail_enrichment(old_value: str, new_value: str) -> bool:
    """Check if new_value is enriching old_value, not contradicting it."""
    old_lower = str(old_value).lower().strip()
    new_lower = str(new_value).lower().strip()

    if old_lower in new_lower:
        return True

    old_words = set(old_lower.split())
    new_words = set(new_lower.split())

    if old_words.issubset(new_words):
        added_words = new_words - old_words
        if added_words & DETAIL_ENRICHMENT_WORDS:
            return True
        if len(added_words) <= 2:
            return True

    return False


def _names_are_related(name1: str, name2: str) -> bool:
    """Simple name relatedness check (stub for monolith's fact_slots.names_are_related)."""
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()
    if n1 == n2:
        return True
    if n1 in n2 or n2 in n1:
        return True
    return False


# ============================================================================
# ML Contradiction Detector
# ============================================================================

class MLContradictionDetector:
    """ML-based contradiction detector.

    Uses trained XGBoost models when available, falls back to heuristic
    detection otherwise.

    Args:
        model_dir: Directory containing trained model files.
            Defaults to ``~/.groundcheck/models/``.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        if model_dir is None:
            model_dir = Path.home() / ".groundcheck" / "models"

        self.model_dir = Path(model_dir)
        self.belief_classifier = None
        self.policy_classifier = None

        if _SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(max_features=100)
            self._load_models()
        else:
            self.vectorizer = None
            logger.info("sklearn not available, using heuristic fallback detection")

    def _load_models(self):
        try:
            try:
                import xgboost  # noqa: F401
            except ImportError:
                logger.info("xgboost not installed, using heuristic fallback")
                return

            belief_path = self.model_dir / "xgboost.pkl"
            if belief_path.exists():
                with open(belief_path, "rb") as f:
                    self.belief_classifier = pickle.load(f)
                logger.info("Loaded belief classifier: %s", belief_path)

            policy_path = self.model_dir / "policy_xgboost.pkl"
            if policy_path.exists():
                with open(policy_path, "rb") as f:
                    self.policy_classifier = pickle.load(f)
                logger.info("Loaded policy classifier: %s", policy_path)

        except Exception as e:
            logger.error("Failed to load ML models: %s", e)

    def check_contradiction(
        self,
        old_value: str,
        new_value: str,
        slot: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Check if new value contradicts old value.

        Returns dict with: ``is_contradiction``, ``category``, ``policy``, ``confidence``.
        """
        if context is None:
            context = {}

        # Transient state check
        if _is_transient_state_value(old_value) or _is_transient_state_value(new_value):
            return {
                "is_contradiction": False,
                "category": "TEMPORAL",
                "policy": "NONE",
                "confidence": 0.65,
                "reason": "transient_state_update",
            }

        # If models not loaded, fall back
        if self.belief_classifier is None or not _SKLEARN_AVAILABLE:
            return self._fallback_detection(old_value, new_value, slot, context)

        # Extract features
        features = self._extract_belief_features(old_value, new_value, context)

        # Retraction pattern check
        query_text = context.get("query", "") or ""
        new_lower = str(new_value).lower()
        query_lower = query_text.lower()

        has_retraction = _has_retraction_pattern(new_lower) or _has_retraction_pattern(query_lower)
        if has_retraction and old_value.lower().strip() != new_value.lower().strip():
            return {
                "is_contradiction": True,
                "category": "CONFLICT",
                "policy": "ASK_USER",
                "confidence": 0.85,
            }

        try:
            category_idx = self.belief_classifier.predict([features])[0]
            categories = ["REFINEMENT", "REVISION", "TEMPORAL", "CONFLICT"]
            category = categories[int(category_idx)]

            proba = self.belief_classifier.predict_proba([features])[0]
            confidence = float(proba.max())

            if _is_semantic_equivalent(old_value, new_value):
                return {
                    "is_contradiction": False,
                    "category": "REFINEMENT",
                    "policy": "NONE",
                    "confidence": confidence,
                }

            if _is_detail_enrichment(old_value, new_value):
                return {
                    "is_contradiction": False,
                    "category": "REFINEMENT",
                    "policy": "NONE",
                    "confidence": confidence,
                }

            # Name relatedness check
            if slot in ("name", "user_name", "spouse_name", "pet_name"):
                if _names_are_related(old_value, new_value):
                    return {
                        "is_contradiction": False,
                        "category": "REFINEMENT",
                        "policy": "NONE",
                        "confidence": confidence,
                    }

            is_contradiction = category in ["REVISION", "CONFLICT"]

            policy = "NONE"
            if is_contradiction and self.policy_classifier is not None:
                policy = self._predict_policy(features, category, slot, context)

            return {
                "is_contradiction": is_contradiction,
                "category": category,
                "policy": policy,
                "confidence": confidence,
            }

        except Exception as e:
            logger.error("ML classification failed: %s", e)
            return self._fallback_detection(old_value, new_value, slot, context)

    def _extract_belief_features(
        self, old_value: str, new_value: str, context: Dict[str, Any],
    ):
        """Extract 18 features for belief classifier."""
        old_value = str(old_value)
        new_value = str(new_value)

        try:
            tfidf = self.vectorizer.fit_transform([old_value, new_value])
            similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        except Exception:
            similarity = 0.0

        old_timestamp = context.get("old_timestamp", time.time() - 86400)
        new_timestamp = context.get("new_timestamp", time.time())
        time_delta_days = (new_timestamp - old_timestamp) / 86400
        recency_score = np.exp(-time_delta_days / 365.0)

        old_lower = old_value.lower()
        new_lower = new_value.lower()

        negation_words = [
            "not", "never", "don't", "doesn't", "didn't", "won't",
            "cannot", "no longer", "n't", "can't",
        ]

        has_retraction = _has_retraction_pattern(new_lower)
        if has_retraction:
            remainder = _extract_remainder_after_retraction(new_lower)
            negation_in_new = int(any(word in remainder for word in negation_words))
        else:
            negation_in_new = int(any(word in new_lower for word in negation_words))

        negation_in_old = int(any(word in old_lower for word in negation_words))
        negation_delta = negation_in_new - negation_in_old

        temporal_words = [
            "now", "currently", "today", "this week", "recently",
            "just", "used to", "previously", "was", "were", "anymore",
        ]
        temporal_in_old = int(any(word in old_lower for word in temporal_words))
        temporal_in_new = int(any(word in new_lower for word in temporal_words))

        correction_words = [
            "actually", "instead", "rather", "changed to",
            "switched to", "i meant", "correction", "wrong", "mistake", "wait",
        ]
        correction_markers = int(any(word in new_lower for word in correction_words))
        if has_retraction:
            correction_markers = 1

        query = context.get("query", new_value)
        query_word_count = len(query.split())
        old_word_count = len(old_value.split())
        new_word_count = len(new_value.split())
        word_count_delta = new_word_count - old_word_count

        memory_confidence = context.get("memory_confidence", 0.85)
        trust_score = context.get("trust_score", 0.85)
        drift_score = 1.0 - similarity
        update_frequency = context.get("update_frequency", 1)
        cross_memory_similarity = context.get("cross_memory_similarity", similarity)

        features = np.array([
            similarity, cross_memory_similarity, time_delta_days,
            recency_score, update_frequency, query_word_count,
            old_word_count, new_word_count, word_count_delta,
            negation_in_new, negation_in_old, negation_delta,
            temporal_in_new, temporal_in_old, correction_markers,
            memory_confidence, trust_score, drift_score,
        ])

        return features

    def _predict_policy(
        self, belief_features, category: str, slot: str, context: Dict[str, Any],
    ) -> str:
        try:
            category_features = np.zeros(4)
            categories = ["REFINEMENT", "REVISION", "TEMPORAL", "CONFLICT"]
            if category in categories:
                category_features[categories.index(category)] = 1

            preference_slots = ["preference", "like", "favorite", "enjoy"]
            is_preference = int(any(p in slot.lower() for p in preference_slots))
            has_correction = belief_features[14] > 0

            policy_features = np.concatenate([
                belief_features,
                category_features[:1],
                [is_preference],
                [int(has_correction)],
            ])

            policy_idx = self.policy_classifier.predict([policy_features])[0]
            policies = ["OVERRIDE", "PRESERVE", "ASK_USER"]
            return policies[int(policy_idx)]

        except Exception as e:
            logger.error("Policy prediction failed: %s", e)
            return "ASK_USER"

    def _fallback_detection(
        self, old_value: Any, new_value: Any, slot: str, context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Heuristic fallback when ML models unavailable."""
        old_value = str(old_value)
        new_value = str(new_value)

        if _is_transient_state_value(old_value) or _is_transient_state_value(new_value):
            return {
                "is_contradiction": False,
                "category": "TEMPORAL",
                "policy": "NONE",
                "confidence": 0.6,
                "reason": "transient_state_update",
            }

        old_lower = old_value.lower()
        new_lower = new_value.lower()

        query_text = context.get("query", "") or ""
        query_lower = query_text.lower()

        negation_words = ["not", "never", "don't", "no longer"]

        has_retraction = _has_retraction_pattern(new_lower) or _has_retraction_pattern(query_lower)
        if has_retraction:
            remainder = _extract_remainder_after_retraction(new_lower)
            has_negation = any(word in remainder for word in negation_words)
        else:
            has_negation = any(word in new_lower for word in negation_words)

        temporal_words = ["now", "currently", "used to", "previously"]
        has_temporal = any(word in new_lower for word in temporal_words)

        normalized_old = old_lower.strip()
        normalized_new = new_lower.strip()
        values_differ = normalized_old != normalized_new

        is_contradiction = values_differ and (has_negation or has_retraction or len(old_value) > 3)

        if has_temporal:
            category = "TEMPORAL"
        elif has_negation:
            category = "CONFLICT"
        elif values_differ:
            category = "REVISION"
        else:
            category = "REFINEMENT"

        policy = "ASK_USER" if is_contradiction else "NONE"

        return {
            "is_contradiction": is_contradiction,
            "category": category,
            "policy": policy,
            "confidence": 0.6,
        }
