"""Neural fact extraction using transformer-based NER."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re

# Lazy imports for optional dependencies
# Global cache: Intentionally shared across instances to avoid reloading heavy models.
# This is standard practice in ML libraries (e.g., transformers, sentence-transformers).
# Thread safety: Pipelines are stateless and safe for concurrent inference.
_ner_pipeline = None
_embedding_model = None

@dataclass
class NeuralExtractionResult:
    """Result from neural extraction."""
    entities: Dict[str, List[str]]  # slot -> values
    confidence: float
    method: str  # "regex", "neural", "hybrid"

class HybridFactExtractor:
    """
    Hybrid fact extraction: fast regex path + neural fallback.
    
    Strategy:
    - Try regex first (<1ms, ~70% recall)
    - If confidence < threshold, use neural NER (10-20ms, ~95% recall)
    - Cache neural model to avoid reload
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.8,
        use_neural: bool = True,
        neural_model: str = "dslim/bert-base-NER"
    ):
        self.confidence_threshold = confidence_threshold
        self.use_neural = use_neural
        self.neural_model_name = neural_model
        self._ner_pipeline = None
        
        # Slot mapping from NER labels to our fact slots
        self.ner_to_slot = {
            "ORG": ["employer", "school", "company"],
            "LOC": ["location", "city"],
            "GPE": ["location", "city"],
            "PER": ["name", "person"],
            "PERSON": ["name", "person"],
        }
    
    def _get_ner_pipeline(self):
        """Lazy load NER pipeline."""
        global _ner_pipeline
        if _ner_pipeline is None and self.use_neural:
            try:
                from transformers import pipeline
                _ner_pipeline = pipeline(
                    "ner",
                    model=self.neural_model_name,
                    aggregation_strategy="simple"
                )
            except ImportError:
                print("Warning: transformers not installed, neural extraction disabled")
                self.use_neural = False
            except Exception as e:
                print(f"Warning: Could not load NER model: {e}")
                self.use_neural = False
        return _ner_pipeline
    
    def _regex_extract_with_confidence(self, text: str) -> Tuple[Dict[str, List[str]], float]:
        """
        Extract facts using regex patterns with confidence estimation.
        
        Confidence is based on:
        - Number of patterns matched
        - Clarity of matches (unambiguous patterns score higher)
        """
        from .fact_extractor import extract_fact_slots
        
        facts = extract_fact_slots(text)
        
        # Convert ExtractedFact objects to simple dict
        result = {}
        for slot, fact in facts.items():
            result[slot] = [str(fact.value)]
        
        # Estimate confidence based on coverage and pattern strength
        if not result:
            confidence = 0.3  # No matches - low confidence
        elif len(result) >= 3:
            confidence = 0.9  # Multiple matches - high confidence
        else:
            confidence = 0.7  # Some matches - medium confidence
        
        return result, confidence
    
    def _neural_extract(self, text: str) -> Dict[str, List[str]]:
        """Extract facts using neural NER."""
        ner = self._get_ner_pipeline()
        if ner is None:
            return {}
        
        try:
            entities = ner(text)
            
            result = {}
            for entity in entities:
                label = entity.get("entity_group", entity.get("entity", ""))
                word = entity.get("word", "").strip()
                score = entity.get("score", 0)
                
                if score < 0.7:  # Skip low-confidence entities
                    continue
                
                # Map NER label to our slots
                slots = self.ner_to_slot.get(label, [])
                for slot in slots:
                    if slot not in result:
                        result[slot] = []
                    if word and word not in result[slot]:
                        result[slot].append(word)
            
            return result
        except Exception as e:
            print(f"Neural extraction error: {e}")
            return {}
    
    def extract(self, text: str) -> NeuralExtractionResult:
        """
        Extract facts using hybrid approach.
        
        Returns:
            NeuralExtractionResult with entities, confidence, and method used
        """
        # Fast path: regex
        regex_facts, confidence = self._regex_extract_with_confidence(text)
        
        if confidence >= self.confidence_threshold or not self.use_neural:
            return NeuralExtractionResult(
                entities=regex_facts,
                confidence=confidence,
                method="regex"
            )
        
        # Slow path: neural
        neural_facts = self._neural_extract(text)
        
        # Merge: neural supplements regex
        merged = dict(regex_facts)
        for slot, values in neural_facts.items():
            if slot not in merged:
                merged[slot] = values
            else:
                # Add new values from neural
                for v in values:
                    if v not in merged[slot]:
                        merged[slot].append(v)
        
        return NeuralExtractionResult(
            entities=merged,
            confidence=0.9,  # Neural path has high confidence
            method="hybrid"
        )
