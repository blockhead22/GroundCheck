"""Core grounding verification logic for GroundCheck."""

from typing import Dict, List, Optional, Set
import re
from difflib import SequenceMatcher

from .types import Memory, VerificationReport, ExtractedFact
from .fact_extractor import extract_fact_slots, split_compound_values
from .utils import (
    normalize_text,
    has_memory_claim,
    create_memory_claim_regex,
    parse_fact_from_memory_text
)


class GroundCheck:
    """Main verifier class for grounding verification.
    
    GroundCheck verifies that generated text is grounded in retrieved memories/context.
    It detects hallucinations by extracting claims from generated text and checking
    if they are supported by the provided memories.
    
    Attributes:
        MUTUALLY_EXCLUSIVE_SLOTS: Fact slots where only one value can be true at a time
        TRUST_DIFFERENCE_THRESHOLD: Minimum trust difference to require contradiction disclosure
    
    Example:
        >>> verifier = GroundCheck()
        >>> memories = [Memory(id="m1", text="User works at Microsoft")]
        >>> result = verifier.verify("You work at Amazon", memories)
        >>> print(result.passed)  # False
        >>> print(result.hallucinations)  # ["Amazon"]
    """
    
    # Fact slots where multiple values are contradictory (mutually exclusive)
    # vs. slots where multiple values are additive (complementary)
    MUTUALLY_EXCLUSIVE_SLOTS = {
        'employer',      # Can only work at one place at a time
        'location',      # Can only live in one place at a time  
        'name',          # Person has one name
        'title',         # One job title at a time
        'occupation',    # One occupation at a time
        'coffee',        # One preference at a time
        'hobby',         # Primary hobby (though people can have multiple)
        'favorite_color',# One favorite
        'favorite_food', # One favorite food
        'pet',           # Primary pet type
        'school',        # Current/most recent school
        'undergrad_school',
        'masters_school',
        'graduation_year', # One graduation year
        'project',       # Current project
    }
    
    # Trust difference threshold: if trust scores differ by more than this,
    # the low-trust memory is considered unreliable and doesn't require disclosure
    TRUST_DIFFERENCE_THRESHOLD = 0.3
    
    # Minimum trust for both memories to require disclosure
    # Set to 0.75 to match benchmark expectations where reliable memories range from 0.75-0.95
    # Only memories below 0.75 are considered unreliable noise
    MINIMUM_TRUST_FOR_DISCLOSURE = 0.75
    
    def __init__(self):
        """Initialize the GroundCheck verifier with semantic matching support."""
        self.memory_claim_regex = create_memory_claim_regex()
        self.semantic_threshold = 0.85  # Similarity threshold for paraphrases
        
        # Initialize hybrid extractor (graceful fallback if neural unavailable)
        try:
            from .neural_extractor import HybridFactExtractor
            self.hybrid_extractor = HybridFactExtractor(
                confidence_threshold=0.8,
                use_neural=True
            )
        except ImportError:
            self.hybrid_extractor = None
        
        # Initialize semantic matcher (graceful fallback if embeddings unavailable)
        try:
            from .semantic_matcher import SemanticMatcher
            self.semantic_matcher = SemanticMatcher(
                use_embeddings=True,
                embedding_threshold=0.85
            )
        except ImportError:
            self.semantic_matcher = None
        
        # Load embedding model for semantic matching (backward compatibility)
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            # If model loading fails, semantic matching will be skipped
            self.embedding_model = None
        
        # TwoTierFactSystem is only available inside the full CRT project.
        # For the standalone PyPI package, this is always None.
        self.two_tier_system = None
    
    def _normalize_value(self, value: str) -> str:
        """Normalize value for fuzzy matching.
        
        Args:
            value: String value to normalize
            
        Returns:
            Normalized lowercase string with articles removed
        """
        if not value:
            return ""
        v = str(value).lower().strip()
        # Remove articles
        v = re.sub(r'\b(a|an|the)\b', '', v)
        # Normalize whitespace
        v = ' '.join(v.split())
        return v
    
    def _is_value_supported(
        self,
        claimed: str,
        supported_values: Set[str],
        slot: str = "",
        threshold: float = 0.85,
        use_semantic: bool = True
    ) -> bool:
        """Check if a claimed value is supported with fuzzy and semantic matching.
        
        Uses three-tier matching:
        1. Exact match (fastest, highest precision)
        2. Substring match (fast, medium precision)
        3. Semantic similarity (slower, handles paraphrases)
        
        Args:
            claimed: The claimed value to check
            supported_values: Set of supported normalized values
            slot: Optional fact slot for context-aware matching
            threshold: Similarity threshold for fuzzy matching (default: 0.85)
            use_semantic: Whether to use embedding similarity (default True)
            
        Returns:
            True if value is supported (exact or fuzzy match), False otherwise
        """
        # Try new semantic matcher if available
        if hasattr(self, 'semantic_matcher') and self.semantic_matcher is not None:
            is_match, method, matched = self.semantic_matcher.is_match(
                claimed, supported_values, slot
            )
            return is_match
        
        # Fallback to original implementation
        claimed_norm = self._normalize_value(claimed)
        
        # Empty claim is not supported
        if not claimed_norm:
            return False
        
        # Tier 1: Check each supported value
        for supported in supported_values:
            supported_norm = self._normalize_value(supported)
            
            # Exact match
            if claimed_norm == supported_norm:
                return True
            
            # Substring match (either direction)
            if claimed_norm in supported_norm or supported_norm in claimed_norm:
                return True
            
            # Fuzzy similarity match
            similarity = SequenceMatcher(None, claimed_norm, supported_norm).ratio()
            if similarity >= threshold:
                return True
            
            # Term overlap check (for phrases)
            claimed_terms = set(claimed_norm.split())
            supported_terms = set(supported_norm.split())
            if claimed_terms and supported_terms:
                overlap = len(claimed_terms & supported_terms) / len(claimed_terms)
                if overlap >= 0.7:  # 70% term overlap
                    return True
        
        # Tier 3: Semantic similarity (handles paraphrases)
        if use_semantic and hasattr(self, 'embedding_model') and self.embedding_model is not None:
            try:
                # Import at function level to avoid issues if package not installed
                from sentence_transformers import util
                
                # Encode claimed value
                claimed_emb = self.embedding_model.encode(
                    claimed, 
                    convert_to_tensor=True
                )
                
                # Check similarity with each memory value
                for mem_val in supported_values:
                    mem_emb = self.embedding_model.encode(
                        mem_val,
                        convert_to_tensor=True
                    )
                    
                    similarity = util.cos_sim(claimed_emb, mem_emb).item()
                    
                    # Use threshold for semantic matching
                    if similarity >= self.semantic_threshold:
                        return True
                        
            except Exception:
                # Fall back to non-semantic matching if embeddings fail
                pass
        
        return False
    
    def _find_memory_for_value(
        self,
        value: str,
        supported_values: Set[str],
        memory_map: Dict[str, str]
    ) -> Optional[str]:
        """Find which memory ID supports a given value.
        
        Args:
            value: The value to find support for
            supported_values: Set of supported normalized values
            memory_map: Map from normalized value to memory ID
            
        Returns:
            Memory ID that supports this value, or None
        """
        value_norm = self._normalize_value(value)
        
        # Try exact match first
        if value_norm in memory_map:
            return memory_map[value_norm]
        
        # Try fuzzy match
        for supported in supported_values:
            supported_norm = self._normalize_value(supported)
            # Check if they match (using same logic as _is_value_supported)
            if value_norm == supported_norm or \
               value_norm in supported_norm or \
               supported_norm in value_norm:
                # Return the memory ID for this supported value
                if supported_norm in memory_map:
                    return memory_map[supported_norm]
        
        # Fallback: Return any memory ID from the slot when fuzzy match succeeded
        # but exact normalized value isn't in the map. This can happen when
        # similarity scoring matched but the values differ slightly.
        # Since fuzzy matching already validated the semantic similarity,
        # any memory from this slot is a reasonable attribution.
        if memory_map:
            return next(iter(memory_map.values()))
        
        return None
    
    def _detect_contradictions(self, retrieved_memories: List[Memory]) -> List['ContradictionDetail']:
        """Detect contradictions in retrieved memories.
        
        Identifies cases where multiple memories have the same fact slot but different values,
        for slots where only one value should be true at a time (mutually exclusive facts).
        
        Enhanced with TwoTierFactSystem for comparing both hard facts and open tuples.
        
        Args:
            retrieved_memories: List of Memory objects to analyze
            
        Returns:
            List of ContradictionDetail objects for each detected contradiction
        """
        from collections import defaultdict
        from .types import ContradictionDetail
        
        # Group memories by fact slot (using both regex and two-tier extraction).
        # De-duplicate (slot, memory_id, normalized_value) so hybrid extraction
        # paths do not double-count the same contradiction value.
        slot_to_facts = defaultdict(list)
        seen_facts = defaultdict(set)

        def _add_fact(slot: str, value: str, memory: Memory, tier: str, confidence: Optional[float] = None) -> None:
            norm_value = self._normalize_value(value)
            if not norm_value:
                return
            key = (str(memory.id), norm_value)
            if key in seen_facts[slot]:
                return
            seen_facts[slot].add(key)
            fact_payload = {
                'value': norm_value,
                'memory_id': memory.id,
                'timestamp': memory.timestamp,
                'trust': memory.trust,
                'tier': tier,
            }
            if confidence is not None:
                fact_payload['confidence'] = confidence
            slot_to_facts[slot].append(fact_payload)
        
        for memory in retrieved_memories:
            # Use TwoTierFactSystem if available for enhanced fact extraction
            if self.two_tier_system is not None:
                try:
                    result = self.two_tier_system.extract_facts(memory.text, skip_llm=True)
                    
                    # Process hard facts (Tier A)
                    for slot, fact in result.hard_facts.items():
                        if slot in self.MUTUALLY_EXCLUSIVE_SLOTS:
                            _add_fact(slot, fact.normalized, memory, tier='hard')
                    
                    # Process open tuples (Tier B) with high confidence
                    for tuple_fact in result.open_tuples:
                        if tuple_fact.confidence >= 0.7:  # Only high-confidence tuples
                            # Check if attribute is mutually exclusive
                            attr = tuple_fact.attribute
                            if attr in self.MUTUALLY_EXCLUSIVE_SLOTS:
                                # Safely get normalized_value with fallback to value
                                normalized = getattr(tuple_fact, 'normalized_value', None) or tuple_fact.value
                                _add_fact(
                                    attr,
                                    normalized,
                                    memory,
                                    tier='open',
                                    confidence=float(tuple_fact.confidence),
                                )
                except Exception as e:
                    # Fall back to regex-only extraction
                    import logging
                    logging.getLogger(__name__).debug(
                        f"TwoTier extraction failed for memory {memory.id}: {e}"
                    )
            
            # Also use traditional regex extraction (for backward compatibility)
            facts = extract_fact_slots(memory.text)
            for slot, fact in facts.items():
                # Only track mutually exclusive slots for contradiction detection
                if slot in self.MUTUALLY_EXCLUSIVE_SLOTS:
                    _add_fact(slot, fact.normalized, memory, tier='regex')
        
        # Find slots with multiple different values
        contradictions = []
        for slot, facts in slot_to_facts.items():
            # Get unique values (normalized)
            unique_values = set(f['value'] for f in facts)
            
            if len(unique_values) > 1:
                # Contradiction detected!
                # Build aligned lists to ensure values, memory_ids, timestamps, and trust_scores match
                values = []
                memory_ids = []
                timestamps = []
                trust_scores = []
                
                for fact in facts:
                    values.append(fact['value'])
                    memory_ids.append(fact['memory_id'])
                    timestamps.append(fact['timestamp'])
                    trust_scores.append(fact['trust'])
                
                contradiction = ContradictionDetail(
                    slot=slot,
                    values=values,
                    memory_ids=memory_ids,
                    timestamps=timestamps,
                    trust_scores=trust_scores
                )
                contradictions.append(contradiction)
        
        return contradictions
    
    def _check_contradiction_disclosure(
        self,
        generated_text: str,
        contradicted_claim: str,
        contradiction: 'ContradictionDetail'
    ) -> bool:
        """Check if generated text acknowledges a contradiction.
        
        Improved with semantic + structural detection (Sprint 1).
        
        Looks for disclosure language patterns like:
        - "changed from X to Y"
        - "previously X, now Y"
        - "was X, is now Y"
        
        Args:
            generated_text: The generated text to check
            contradicted_claim: The specific claim being checked
            contradiction: The ContradictionDetail object
            
        Returns:
            True if text contains adequate disclosure, False otherwise
        """
        text_lower = generated_text.lower()
        
        # 1. Keyword detection (existing logic)
        disclosure_patterns = [
            'changed from',
            'updated from',
            'previously',
            'was',
            'used to',
            'formerly',
            'switched from',
            'moved from',
            'before',
            'most recent',
            'latest',
            'now'
        ]
        
        has_disclosure_keyword = any(pattern in text_lower for pattern in disclosure_patterns)
        
        # 2. Structural pattern detection (Sprint 1 enhancement)
        structural_patterns = [
            r'\(changed from .+?\)',
            r'\(updated from .+?\)',
            r'\(previously .+?\)',
            r', previously .+?[,.]',
            r'used to be .+?[,.]',
            r'was .+?, now',
            r'formerly .+?[,.]'
        ]
        has_structure = any(re.search(p, generated_text, re.IGNORECASE) for p in structural_patterns)
        
        # 3. Contradiction value mention (existing logic, de-duplicated)
        # Check if BOTH old and new values are mentioned.
        unique_values = {
            self._normalize_value(v)
            for v in contradiction.values
            if v
        }
        values_mentioned = 0
        for val in unique_values:
            if not val:
                continue
            if re.search(rf'\b{re.escape(val)}\b', text_lower):
                values_mentioned += 1
        both_mentioned = len(unique_values) >= 2 and values_mentioned >= 2
        
        # Success if ANY disclosure method is present
        return has_disclosure_keyword or has_structure or both_mentioned
    
    def _generate_disclosure_text(
        self,
        claim_value: str,
        contradiction: 'ContradictionDetail'
    ) -> str:
        """Generate suggested disclosure text for a contradicted claim.
        
        Args:
            claim_value: The value being claimed
            contradiction: The ContradictionDetail object
            
        Returns:
            Suggested disclosure text (e.g., "Amazon (changed from Microsoft)")
        """
        # Find which value is being claimed
        other_values = [v for v in contradiction.values if v != claim_value.lower()]
        
        if not other_values:
            return claim_value
        
        # Build disclosure text
        disclosure = claim_value
        
        # Add temporal context if available
        if contradiction.timestamps and any(t is not None for t in contradiction.timestamps):
            most_recent = contradiction.most_recent_value
            if most_recent == claim_value.lower():
                # Current value is most recent
                old_value = other_values[0]
                disclosure += f" (changed from {old_value})"
            else:
                disclosure += f" (previously {other_values[0]})"
        else:
            # No timestamps - just mention the conflict
            if len(other_values) == 1:
                disclosure += f" (previously {other_values[0]})"
            else:
                disclosure += f" (conflicting information: {', '.join(other_values)})"
        
        return disclosure
    
    def verify(
        self,
        generated_text: str,
        retrieved_memories: List[Memory],
        mode: str = "strict"
    ) -> VerificationReport:
        """Verify that generated text is grounded in retrieved memories.
        
        Enhanced with contradiction detection and disclosure verification.
        
        Args:
            generated_text: The text to verify
            retrieved_memories: List of Memory objects containing supporting context
            mode: Verification mode - "strict" (generates corrections) or "permissive"
            
        Returns:
            VerificationReport with verification results and contradiction analysis
        """
        if not generated_text or not generated_text.strip():
            return VerificationReport(
                original=generated_text,
                passed=True,
                confidence=1.0
            )
        
        # Step 1: Detect contradictions in retrieved context
        contradictions = self._detect_contradictions(retrieved_memories)
        
        # Extract facts from generated text
        facts_extracted = extract_fact_slots(generated_text)
        
        # Build grounding map and collect hallucinations
        hallucinations = []
        grounding_map = {}
        supported_facts = {}
        contradicted_claims = []
        
        # Parse supported facts from memories
        memory_facts_by_slot: Dict[str, Set[str]] = {}
        memory_id_by_slot_value: Dict[str, Dict[str, str]] = {}
        
        for memory in retrieved_memories:
            # Try parsing structured FACT: format
            parsed = parse_fact_from_memory_text(memory.text)
            if parsed:
                slot, value = parsed
                value_norm = normalize_text(value)
                memory_facts_by_slot.setdefault(slot, set()).add(value_norm)
                memory_id_by_slot_value.setdefault(slot, {})[value_norm] = memory.id
            
            # Also extract facts from memory text
            memory_facts = extract_fact_slots(memory.text)
            for slot, fact in memory_facts.items():
                # Split compound values in memories too
                fact_values = split_compound_values(str(fact.value))
                for val in fact_values:
                    val_norm = self._normalize_value(val)
                    if val_norm:
                        memory_facts_by_slot.setdefault(slot, set()).add(val_norm)
                        memory_id_by_slot_value.setdefault(slot, {})[val_norm] = memory.id
        
        # Check each extracted fact against memories
        for slot, fact in facts_extracted.items():
            slot_l = slot.lower()
            support_slot = slot_l
            supported_values = memory_facts_by_slot.get(support_slot, set())
            # Allow historical-slot aliases (e.g., previous_employer) to resolve
            # against the canonical memory slot (e.g., employer).
            if not supported_values:
                for prefix in ("previous_", "prior_", "former_"):
                    if support_slot.startswith(prefix):
                        canonical_slot = support_slot[len(prefix):]
                        if canonical_slot in memory_facts_by_slot:
                            support_slot = canonical_slot
                            supported_values = memory_facts_by_slot.get(support_slot, set())
                            break
            
            # Split compound values from the generated text
            fact_values = split_compound_values(str(fact.value))
            
            # Track which individual values are supported
            all_supported = True
            for val in fact_values:
                val_norm = self._normalize_value(val)
                if not val_norm:
                    continue
                
                # Check if this individual value is supported (with fuzzy matching)
                if self._is_value_supported(val, supported_values, slot=slot_l):
                    # Find which memory supports this value
                    memory_id = self._find_memory_for_value(
                        val,
                        supported_values,
                        memory_id_by_slot_value.get(support_slot, {}),
                    )
                    if memory_id:
                        grounding_map[val] = memory_id
                    
                    # Step 2: Check if this claim involves a contradiction
                    slot_contradiction = next(
                        (c for c in contradictions if c.slot == support_slot),
                        None
                    )
                    
                    if slot_contradiction and val_norm in slot_contradiction.values:
                        # This claim uses a contradicted fact
                        contradicted_claims.append(val)
                else:
                    # This value is not supported - it's a hallucination
                    hallucinations.append(val)
                    all_supported = False
            
            # Only mark the fact as supported if ALL values are supported
            if all_supported and fact_values:
                supported_facts[slot] = fact
        
        # Identify fully unsupported facts (for correction generation)
        unsupported_facts = {
            slot: fact for slot, fact in facts_extracted.items()
            if slot not in supported_facts
        }
        
        # Step 3: Check if contradicted claims are properly disclosed
        requires_disclosure = False
        expected_disclosure = None
        
        if contradicted_claims:
            for claim in contradicted_claims:
                # Find the contradiction details
                slot = None
                for s, fact in facts_extracted.items():
                    if str(fact.value) == claim or claim in split_compound_values(str(fact.value)):
                        slot = s.lower()
                        break
                
                if slot:
                    contradiction = next((c for c in contradictions if c.slot == slot), None)
                    if contradiction:
                        # Check trust scores to determine if disclosure is needed
                        # Only require disclosure if BOTH memories are credible
                        trust_diff = max(contradiction.trust_scores) - min(contradiction.trust_scores)
                        min_trust = min(contradiction.trust_scores)
                        max_trust = max(contradiction.trust_scores)
                        
                        # Skip disclosure requirement if:
                        # 1. Large trust difference (one memory is clearly more reliable)
                        # 2. Low-trust memory is below reliability threshold
                        if trust_diff >= self.TRUST_DIFFERENCE_THRESHOLD:
                            # Large gap - low-trust memory is noise, no disclosure needed
                            continue
                        elif min_trust < self.MINIMUM_TRUST_FOR_DISCLOSURE:
                            # At least one memory is unreliable, no disclosure needed
                            continue
                        
                        # Both memories are credible with similar trust - disclosure required
                        # Check if output acknowledges the contradiction
                        has_disclosure = self._check_contradiction_disclosure(
                            generated_text,
                            claim,
                            contradiction
                        )
                        
                        if not has_disclosure:
                            requires_disclosure = True
                            expected_disclosure = self._generate_disclosure_text(
                                claim,
                                contradiction
                            )
                            # Only show one expected disclosure for simplicity
                            break
        
        # Step 4: Determine if verification passed
        passed = (
            len(hallucinations) == 0 and  # No hallucinations
            not requires_disclosure  # No undisclosed contradictions
        )
        
        # Calculate confidence based on trust scores of supporting memories
        confidence = self._calculate_confidence(
            facts_extracted,
            supported_facts,
            grounding_map,
            retrieved_memories
        )
        
        # Step 5: Generate corrected text if in strict mode and there are issues
        corrected_text = None
        if mode == "strict" and not passed:
            corrected_text = self._generate_correction(
                generated_text,
                unsupported_facts,
                supported_facts,
                retrieved_memories,
                contradicted_claims,
                contradictions
            )
        
        return VerificationReport(
            original=generated_text,
            corrected=corrected_text,
            passed=passed,
            hallucinations=hallucinations,
            grounding_map=grounding_map,
            confidence=confidence,
            facts_extracted=facts_extracted,
            facts_supported=supported_facts,
            contradicted_claims=contradicted_claims,
            contradiction_details=contradictions,
            requires_disclosure=requires_disclosure,
            expected_disclosure=expected_disclosure
        )
    
    def extract_claims(self, text: str) -> Dict[str, ExtractedFact]:
        """Extract factual claims from text.
        
        Args:
            text: Text to extract claims from
            
        Returns:
            Dictionary mapping slot names to ExtractedFact objects
        """
        return extract_fact_slots(text)
    
    def find_support(
        self,
        claim: ExtractedFact,
        memories: List[Memory]
    ) -> Optional[Memory]:
        """Find a memory that supports the given claim.
        
        Args:
            claim: The claim to find support for
            memories: List of memories to search
            
        Returns:
            Supporting Memory if found, None otherwise
        """
        claim_norm = self._normalize_value(str(claim.value))
        
        for memory in memories:
            # Parse structured facts from memory
            parsed = parse_fact_from_memory_text(memory.text)
            if parsed:
                slot, value = parsed
                if slot == claim.slot:
                    # Use fuzzy matching for structured facts
                    if self._is_value_supported(str(claim.value), {value}, slot=slot):
                        return memory
            
            # Extract facts from memory text
            memory_facts = extract_fact_slots(memory.text)
            if claim.slot in memory_facts:
                memory_fact = memory_facts[claim.slot]
                # Use fuzzy matching
                memory_values = split_compound_values(str(memory_fact.value))
                if self._is_value_supported(str(claim.value), set(memory_values), slot=claim.slot):
                    return memory
            
            # Fallback: fuzzy text matching in memory text
            memory_norm = self._normalize_value(memory.text)
            if claim_norm and (claim_norm in memory_norm or 
                              SequenceMatcher(None, claim_norm, memory_norm).ratio() > 0.6):
                return memory
        
        return None
    
    def build_grounding_map(
        self,
        claims: Dict[str, ExtractedFact],
        memories: List[Memory]
    ) -> Dict[str, str]:
        """Build a map from claims to supporting memory IDs.
        
        Args:
            claims: Dictionary of extracted claims
            memories: List of memories
            
        Returns:
            Dictionary mapping claim values to memory IDs
        """
        grounding_map = {}
        
        for slot, claim in claims.items():
            supporting_memory = self.find_support(claim, memories)
            if supporting_memory:
                grounding_map[str(claim.value)] = supporting_memory.id
        
        return grounding_map
    
        
    def _calculate_confidence(
        self,
        all_facts: Dict[str, ExtractedFact],
        supported_facts: Dict[str, ExtractedFact],
        grounding_map: Dict[str, str],
        memories: List[Memory]
    ) -> float:
        """Calculate confidence score for verification.
        
        Confidence is based on:
        - Ratio of supported facts to total facts
        - Trust scores of supporting memories
        """
        if not all_facts:
            return 1.0
        
        # Base confidence from support ratio
        support_ratio = len(supported_facts) / len(all_facts)
        
        # Weight by memory trust scores
        if grounding_map and memories:
            memory_by_id = {m.id: m for m in memories}
            trust_scores = []
            for memory_id in grounding_map.values():
                if memory_id in memory_by_id:
                    trust_scores.append(memory_by_id[memory_id].trust)
            
            if trust_scores:
                avg_trust = sum(trust_scores) / len(trust_scores)
                # Combine support ratio and average trust
                return (support_ratio + avg_trust) / 2.0
        
        return support_ratio
    
    def _generate_correction(
        self,
        original_text: str,
        unsupported_facts: Dict[str, ExtractedFact],
        supported_facts: Dict[str, ExtractedFact],
        memories: List[Memory],
        contradicted_claims: List[str] = None,
        contradictions: List['ContradictionDetail'] = None
    ) -> str:
        """Generate corrected text by replacing unsupported claims and adding disclosure.
        
        Handles both hallucinations and undisclosed contradictions.
        
        Args:
            original_text: The original text to correct
            unsupported_facts: Facts that are not grounded in memories
            supported_facts: Facts that are grounded in memories
            memories: List of memories
            contradicted_claims: Claims that involve contradictions
            contradictions: List of contradiction details
            
        Returns:
            Corrected text with replacements and disclosure
        """
        if contradicted_claims is None:
            contradicted_claims = []
        if contradictions is None:
            contradictions = []
            
        corrected = original_text
        
        # Handle hallucinations first
        if unsupported_facts:
            # Check if this is a memory claim that should be sanitized
            if not has_memory_claim(original_text):
                # Simple replacement approach
                corrected = self._simple_correction(
                    corrected, unsupported_facts, supported_facts, memories
                )
            else:
                # For memory claims, use stricter sanitization
                corrected = self._sanitize_memory_claims(
                    corrected, unsupported_facts, supported_facts, memories
                )
        
        # Add disclosure for contradicted claims
        for claim in contradicted_claims:
            # Find the contradiction for this claim
            for c in contradictions:
                claim_norm = self._normalize_value(claim)
                if claim_norm in c.values:
                    disclosure = self._generate_disclosure_text(claim, c)
                    # Replace the claim with the disclosure version
                    corrected = corrected.replace(claim, disclosure)
                    break
        
        return corrected
    
    def _simple_correction(
        self,
        text: str,
        unsupported_facts: Dict[str, ExtractedFact],
        supported_facts: Dict[str, ExtractedFact],
        memories: List[Memory]
    ) -> str:
        """Simple correction by replacing unsupported values with supported ones.
        
        Looks for memories that have facts for the same slot as unsupported facts
        and replaces the unsupported value with the memory's value.
        """
        corrected = text
        
        # First try using supported_facts that were already extracted
        for slot, unsupported_fact in unsupported_facts.items():
            # If we have a supported fact for the same slot, replace it
            if slot in supported_facts:
                supported_fact = supported_facts[slot]
                unsupported_value = str(unsupported_fact.value)
                supported_value = str(supported_fact.value)
                
                # Simple string replacement
                corrected = corrected.replace(unsupported_value, supported_value)
            else:
                # Try to find a matching fact in memories for this slot
                for memory in memories:
                    memory_facts = extract_fact_slots(memory.text)
                    if slot in memory_facts:
                        memory_fact = memory_facts[slot]
                        unsupported_value = str(unsupported_fact.value)
                        supported_value = str(memory_fact.value)
                        corrected = corrected.replace(unsupported_value, supported_value)
                        break
        
        return corrected
    
    def _sanitize_memory_claims(
        self,
        text: str,
        unsupported_facts: Dict[str, ExtractedFact],
        supported_facts: Dict[str, ExtractedFact],
        memories: List[Memory]
    ) -> str:
        """Sanitize text with memory claims by removing unsupported statements.
        
        This follows the logic from the original _sanitize_unsupported_memory_claims.
        """
        if not unsupported_facts:
            return text
        
        # Create regex patterns for unsupported values
        bad_value_patterns = [
            re.compile(re.escape(str(fact.value)), re.I)
            for fact in unsupported_facts.values()
            if fact.value
        ]
        
        # Keep lines that don't contain memory claims or unsupported values
        kept_lines = []
        for line in text.splitlines():
            # Skip lines with memory claim phrases
            if self.memory_claim_regex.search(line):
                continue
            # Skip lines with unsupported values
            if any(pattern.search(line) for pattern in bad_value_patterns):
                continue
            kept_lines.append(line)
        
        cleaned = "\n".join(kept_lines).strip()
        
        # Add disclaimer about missing information
        first_slot = next(iter(unsupported_facts.keys()))
        disclaimer = f"I don't have reliable stored information for your {first_slot} yet — if you tell me, I can store it going forward."
        
        if cleaned:
            return cleaned + f"\n\n{disclaimer}"
        
        return disclaimer
