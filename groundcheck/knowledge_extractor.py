"""Knowledge-based fact extraction using verb ontology + entity taxonomy.

Tier 1.5: Sits between regex extraction (Tier 1) and LLM extraction (Tier 3).
Uses curated knowledge shipped as JSON — zero dependencies, deterministic,
but understands concepts like "migrated to Postgres" = database: PostgreSQL.

Strategy:
    1. Decompose text into clauses (sentence splitting + conjunction splitting)
    2. For each clause, identify recognized entities via taxonomy lookup
    3. For each clause, identify verb semantics via ontology lookup
    4. Apply inference rules: verb_category + entity_category → extracted fact
    5. Assign confidence based on verb category (tentative = 0.35, adoption = 0.90)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .types import ExtractedFact


# ── Data loading (done once at module import, ~2ms) ──────────────────────────

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load_json(filename: str) -> dict:
    path = os.path.join(_DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Lazy-loaded singletons
_verb_ontology: Optional[dict] = None
_entity_taxonomy: Optional[dict] = None
_entity_lookup: Optional[Dict[str, Tuple[str, str, bool]]] = None
_verb_lookup: Optional[List[Tuple[str, str, float, str]]] = None


def _get_verb_ontology() -> dict:
    global _verb_ontology
    if _verb_ontology is None:
        _verb_ontology = _load_json("verb_ontology.json")
    return _verb_ontology


def _get_entity_taxonomy() -> dict:
    global _entity_taxonomy
    if _entity_taxonomy is None:
        _entity_taxonomy = _load_json("entity_taxonomy.json")
    return _entity_taxonomy


def _get_entity_lookup() -> Dict[str, Tuple[str, str, bool]]:
    """Build a lowercase entity → (canonical_name, slot, exclusive) lookup.

    Returns dict mapping lowercase entity names to their metadata.
    """
    global _entity_lookup
    if _entity_lookup is not None:
        return _entity_lookup

    taxonomy = _get_entity_taxonomy()
    _entity_lookup = {}
    for category_key, category in taxonomy.items():
        if category_key == "_meta":
            continue
        slot = category["slot"]
        exclusive = category.get("exclusive", True)
        for entity in category["entities"]:
            _entity_lookup[entity.lower()] = (entity, slot, exclusive)
    return _entity_lookup


def _get_verb_lookup() -> List[Tuple[str, str, float, str]]:
    """Build a list of (verb_phrase, category, confidence, temporal) sorted longest-first.

    Longest-first ensures "migrated to" matches before "migrated".
    """
    global _verb_lookup
    if _verb_lookup is not None:
        return _verb_lookup

    ontology = _get_verb_ontology()
    entries = []
    for category_key, category in ontology.items():
        if category_key == "_meta":
            continue
        confidence = category.get("confidence", 0.80)
        temporal = category.get("temporal", "current")
        for verb in category["verbs"]:
            entries.append((verb.lower(), category_key, confidence, temporal))

    # Sort by verb length descending so multi-word verbs match first
    entries.sort(key=lambda x: len(x[0]), reverse=True)
    _verb_lookup = entries
    return _verb_lookup


# ── Clause decomposition ─────────────────────────────────────────────────────

# Clause boundary patterns
_CLAUSE_SPLITTERS = re.compile(
    r"(?:\.\s+)"          # Period + space (sentence boundary)
    r"|(?:;\s*)"          # Semicolon
    r"|(?:,\s*(?:and|but|so|then|also|plus)\s+)"  # Comma + conjunction
    r"|(?:\s+(?:but|however|although|though|except)\s+)"  # Contrastive conjunctions (always split)
    r"|(?:\s+after\s+)"   # "after" as clause boundary (temporal)
    r"|(?:\s+before\s+)"  # "before" as clause boundary (temporal)
    r"|(?:\s+since\s+)"   # "since" as clause boundary (temporal/causal)
    r"|(?:\n+)",          # Newlines
    re.IGNORECASE
)

# "and" is special: only split on it when it separates independent clauses,
# not when it's part of a list like "X and Y for Z"
_AND_SPLITTER = re.compile(r",\s+", re.IGNORECASE)

# Skip patterns — don't extract from questions, hedges, negated hypotheticals
_SKIP_PATTERNS = re.compile(
    r"^\s*(?:"
    r"(?:what|why|how|when|where|which|who|whose|whom)\s"  # Questions
    r"|(?:i\s+(?:don'?t|do\s+not)\s+(?:know|think|remember|recall))"  # Uncertainty
    r"|(?:not\s+sure\s+(?:if|about|whether))"  # Hedging 
    r"|(?:i\s+wonder)"  # Wondering
    r"|(?:maybe\s+we\s+should)"  # Weak suggestions
    r")",
    re.IGNORECASE
)

# Negative context words — when near an entity, suppress positive extraction
# or infer deprecation/negative sentiment
_NEGATIVE_CONTEXT = re.compile(
    r"\b(?:disaster|nightmare|mess|terrible|horrible|awful|broken|"
    r"fiasco|catastrophe|problem|problems|issue|issues|trouble|troubles|"
    r"pain|painful|buggy|unstable|unreliable|slow|bloated)\b",
    re.IGNORECASE
)


def split_into_clauses(text: str) -> List[str]:
    """Split text into individual clauses for extraction.

    Handles sentence boundaries, conjunctions, semicolons, newlines.
    Also splits on bare commas when they separate independent clauses.
    """
    # First pass: structural splits (periods, semicolons, conjunctions, temporal)
    clauses = _CLAUSE_SPLITTERS.split(text)

    # Second pass: split on bare commas (independent clause separator)
    expanded = []
    for clause in clauses:
        parts = _AND_SPLITTER.split(clause)
        expanded.extend(parts)

    result = []
    for clause in expanded:
        clause = clause.strip()
        if len(clause) >= 5:  # Skip tiny fragments
            result.append(clause)
    return result if result else [text.strip()]


# ── Entity recognition ────────────────────────────────────────────────────────

@dataclass
class RecognizedEntity:
    """An entity found in text via taxonomy lookup."""
    canonical: str       # Canonical name from taxonomy (e.g., "PostgreSQL")
    slot: str           # Inferred slot (e.g., "database")
    exclusive: bool     # Whether this slot is mutually exclusive
    start: int          # Character position in clause
    end: int            # Character position end


def find_entities(text: str) -> List[RecognizedEntity]:
    """Find all taxonomy-recognized entities in text.

    Uses word-boundary-aware matching, longest match first.
    """
    lookup = _get_entity_lookup()
    text_lower = text.lower()
    found: List[RecognizedEntity] = []
    used_spans: List[Tuple[int, int]] = []

    # Sort by entity length descending to prefer longer matches
    # e.g., "Ruby on Rails" before "Ruby"
    sorted_entities = sorted(lookup.keys(), key=len, reverse=True)

    for entity_lower in sorted_entities:
        # Try to find this entity in the text
        idx = 0
        while True:
            pos = text_lower.find(entity_lower, idx)
            if pos == -1:
                break

            end = pos + len(entity_lower)

            # Check this span doesn't overlap with already-found entities
            overlaps = any(
                not (end <= us or pos >= ue)
                for us, ue in used_spans
            )
            if overlaps:
                idx = pos + 1
                continue

            # Word boundary check: don't match "Rustacean" for "Rust"
            before_ok = (pos == 0 or not text_lower[pos - 1].isalnum())
            after_ok = (end == len(text_lower) or not text_lower[end].isalnum())

            if before_ok and after_ok:
                canonical, slot, exclusive = lookup[entity_lower]
                found.append(RecognizedEntity(
                    canonical=canonical,
                    slot=slot,
                    exclusive=exclusive,
                    start=pos,
                    end=end,
                ))
                used_spans.append((pos, end))

            idx = pos + 1

    return found


# ── Verb recognition ──────────────────────────────────────────────────────────

@dataclass
class RecognizedVerb:
    """A verb/phrase found in text via ontology lookup."""
    phrase: str         # The matched verb phrase
    category: str       # Semantic category (adoption, migration, etc.)
    confidence: float   # Base confidence for this verb type
    temporal: str       # Temporal implication (current, past, future, transition)
    start: int          # Character position in clause
    end: int            # Character position end


def find_verbs(text: str) -> List[RecognizedVerb]:
    """Find all ontology-recognized verbs in text.

    Longest match first — "switched to" matches before "switched".
    """
    verb_list = _get_verb_lookup()
    text_lower = text.lower()
    found: List[RecognizedVerb] = []
    used_spans: List[Tuple[int, int]] = []

    for verb_phrase, category, confidence, temporal in verb_list:
        idx = 0
        while True:
            pos = text_lower.find(verb_phrase, idx)
            if pos == -1:
                break

            end = pos + len(verb_phrase)

            # Check span doesn't overlap
            overlaps = any(
                not (end <= us or pos >= ue)
                for us, ue in used_spans
            )
            if overlaps:
                idx = pos + 1
                continue

            # Word boundary check
            before_ok = (pos == 0 or not text_lower[pos - 1].isalnum())
            after_ok = (end == len(text_lower) or not text_lower[end].isalnum())

            if before_ok and after_ok:
                found.append(RecognizedVerb(
                    phrase=verb_phrase,
                    category=category,
                    confidence=confidence,
                    temporal=temporal,
                    start=pos,
                    end=end,
                ))
                used_spans.append((pos, end))

            idx = pos + 1

    return found


# ── Inference engine ──────────────────────────────────────────────────────────

@dataclass
class KnowledgeFact:
    """A fact inferred from knowledge-based extraction."""
    slot: str             # e.g., "database"
    value: str            # e.g., "PostgreSQL"
    confidence: float     # 0.0 - 1.0, from verb category
    verb_category: str    # e.g., "migration", "adoption"
    temporal: str         # "current", "past", "future", "transition"
    clause: str           # The source clause
    deprecated_value: Optional[str] = None  # For migration verbs: the old value


def infer_facts(text: str) -> List[KnowledgeFact]:
    """Extract facts by combining verb semantics with entity recognition.

    This is the main inference engine. For each clause:
    1. Find entities (what things are mentioned)
    2. Find verbs (what's being done)
    3. Combine: verb + entity = fact

    Special handling:
    - Migration verbs: look for FROM entity and TO entity
    - Tentative verbs: low confidence, don't overwrite existing facts
    - Deprecation verbs: mark entity as no longer active
    - Verb inheritance: verbless clauses inherit from previous clause
    - Deduplication: migration suppresses standalone deprecation/adoption
    - Tentative override: tentative verb before migration → whole clause is tentative
    """
    all_facts: List[KnowledgeFact] = []

    clauses = split_into_clauses(text)
    prev_verb: Optional[RecognizedVerb] = None  # For verb context inheritance

    for clause in clauses:
        # Skip questions, hedges, uncertainty
        if _SKIP_PATTERNS.match(clause):
            prev_verb = None
            continue

        entities = find_entities(clause)
        verbs = find_verbs(clause)

        # CRITICAL: Remove entities whose spans overlap with verb spans
        # This prevents "go" in "go with" matching as the Go language
        if verbs and entities:
            verb_spans = [(v.start, v.end) for v in verbs]
            entities = [
                e for e in entities
                if not any(
                    not (e.end <= vs or e.start >= ve)
                    for vs, ve in verb_spans
                )
            ]

        if not entities:
            # Update prev_verb if this clause has verbs (for inheritance)
            if verbs:
                prev_verb = verbs[0]
            continue

        # If no recognized verb, try copular or inherit from previous clause
        if not verbs:
            # Check for negative context — suppress extraction
            if _NEGATIVE_CONTEXT.search(clause):
                # Negative context without explicit verb → treat as deprecation hint
                for entity in entities:
                    all_facts.append(KnowledgeFact(
                        slot=entity.slot,
                        value=entity.canonical,
                        confidence=0.60,
                        verb_category="deprecation",
                        temporal="past",
                        clause=clause,
                    ))
                prev_verb = None
                continue

            # Check for copular patterns: "our database is X", "the backend is X"
            copular = _try_copular_inference(clause, entities)
            if copular:
                all_facts.extend(copular)
            elif prev_verb is not None:
                # Inherit verb context from previous clause
                # e.g., "We use X for CI and Y for monitoring"
                facts = _infer_standard(prev_verb, entities)
                all_facts.extend(facts)
            else:
                # Bare entity mention without verb → weak assertion at low confidence
                # e.g., "TypeScript everywhere" = they use TypeScript
                for entity in entities:
                    all_facts.append(KnowledgeFact(
                        slot=entity.slot,
                        value=entity.canonical,
                        confidence=0.50,
                        verb_category="assertion",
                        temporal="current",
                        clause=clause,
                    ))
            continue

        # Check for tentative override: if tentative verb appears alongside
        # migration/adoption verb, the whole thing is tentative
        # e.g., "considering switching to Rust"
        has_tentative = any(v.category == "tentative" for v in verbs)
        non_tentative = [v for v in verbs if v.category != "tentative"]

        if has_tentative and non_tentative:
            # Tentative overrides — use the non-tentative verb's mechanics
            # but with tentative confidence and temporal
            tentative_verb_info = next(v for v in verbs if v.category == "tentative")
            for verb in non_tentative:
                # Create a modified verb with tentative semantics
                override = RecognizedVerb(
                    phrase=verb.phrase,
                    category=verb.category,
                    confidence=tentative_verb_info.confidence,
                    temporal=tentative_verb_info.temporal,
                    start=verb.start,
                    end=verb.end,
                )
                if verb.category == "migration":
                    facts = _infer_migration(clause, override, entities)
                else:
                    facts = _infer_standard(override, entities)
                # Mark all as tentative
                for f in facts:
                    f.verb_category = "tentative"
                all_facts.extend(facts)
        else:
            # Normal processing: if multiple verbs, assign each entity
            # to its nearest verb (position-aware matching)
            clause_facts: List[KnowledgeFact] = []

            if len(verbs) > 1:
                # Position-aware: assign each entity to nearest verb
                verb_entity_map: Dict[int, List[RecognizedEntity]] = {
                    i: [] for i in range(len(verbs))
                }
                for entity in entities:
                    # Measure distance from entity to each verb
                    best_vi = 0
                    best_dist = float("inf")
                    for vi, verb in enumerate(verbs):
                        # Distance = gap between entity and verb spans
                        if entity.start >= verb.end:
                            dist = entity.start - verb.end
                        elif entity.end <= verb.start:
                            dist = verb.start - entity.end
                        else:
                            dist = 0  # Overlapping
                        if dist < best_dist:
                            best_dist = dist
                            best_vi = vi
                    verb_entity_map[best_vi].append(entity)

                for vi, verb in enumerate(verbs):
                    ents = verb_entity_map[vi]
                    if not ents:
                        continue
                    if verb.category == "migration":
                        facts = _infer_migration(clause, verb, ents)
                    elif verb.category == "deprecation":
                        facts = _infer_deprecation(verb, ents, clause)
                    else:
                        facts = _infer_standard(verb, ents)
                    clause_facts.extend(facts)
            else:
                verb = verbs[0]
                if verb.category == "migration":
                    facts = _infer_migration(clause, verb, entities)
                elif verb.category == "deprecation":
                    facts = _infer_deprecation(verb, entities, clause)
                else:
                    facts = _infer_standard(verb, entities)
                clause_facts.extend(facts)

            # Deduplication: if migration covers an entity, remove
            # standalone deprecation/adoption for the same entity
            clause_facts = _deduplicate_facts(clause_facts)
            all_facts.extend(clause_facts)

        # Track last verb for inheritance
        prev_verb = verbs[0] if verbs else prev_verb

    return all_facts


def _deduplicate_facts(facts: List[KnowledgeFact]) -> List[KnowledgeFact]:
    """Remove redundant facts within a single clause.

    Rules:
    - If migration covers entity X as deprecated_value, remove standalone deprecation for X
    - If migration covers entity Y as target, remove standalone adoption for Y
    - Prefer migration facts over standalone adoption/deprecation
    """
    # Find entities covered by migration facts
    migration_targets: Set[str] = set()
    migration_deprecated: Set[str] = set()
    for f in facts:
        if f.verb_category == "migration":
            migration_targets.add(f.value)
            if f.deprecated_value:
                migration_deprecated.add(f.deprecated_value)

    if not migration_targets and not migration_deprecated:
        return facts

    result = []
    for f in facts:
        if f.verb_category == "migration":
            result.append(f)
            continue
        # Skip deprecation if migration already covers this entity
        if f.verb_category == "deprecation" and f.value in migration_deprecated:
            continue
        if f.verb_category == "deprecation" and f.value in migration_targets:
            continue
        # Skip adoption if migration already covers this entity as target
        if f.verb_category == "adoption" and f.value in migration_targets:
            continue
        result.append(f)
    return result


def _try_copular_inference(
    clause: str, entities: List[RecognizedEntity]
) -> List[KnowledgeFact]:
    """Try to infer facts from copular/possessive patterns without recognized verbs.

    Handles: "our database is Postgres", "the backend is FastAPI", etc.
    For entities not immediately after the copular verb, use their taxonomy slot.
    """
    facts = []
    clause_lower = clause.lower()

    # Look for "SLOT_WORD is/are ENTITY" patterns
    slot_hints = {
        "database": ["database", "db", "data store", "datastore"],
        "backend_framework": ["backend", "server", "api server", "web server"],
        "frontend_framework": ["frontend", "front-end", "client", "ui framework"],
        "language": ["language", "lang", "programming language"],
        "cloud_provider": ["cloud", "hosting", "infrastructure", "infra"],
        "orchestration": ["orchestration", "container orchestration"],
        "ci_cd": ["ci", "cd", "ci/cd", "pipeline", "build system"],
        "message_queue": ["queue", "message queue", "message broker", "broker"],
        "monitoring": ["monitoring", "observability", "alerting", "logging"],
        "os": ["os", "operating system", "distro", "distribution"],
        "auth": ["auth", "authentication", "identity", "sso"],
        "editor": ["editor", "ide"],
        "testing": ["testing", "test framework", "test runner"],
        "vcs": ["repo", "repository", "version control"],
        "api_style": ["api", "api style"],
        "package_manager": ["package manager"],
    }

    # Find copular verb position
    copular_match = re.search(
        r"\b(?:is|are|was|will\s+be)\b", clause_lower
    )

    # Track which entities have been matched via copular hint
    matched_entities: Set[int] = set()

    if copular_match:
        cop_end = copular_match.end()

        # Sort entities by proximity to copular verb (nearest first)
        entities_after_cop = sorted(
            [(i, e) for i, e in enumerate(entities) if e.start >= cop_end - 3],
            key=lambda x: x[1].start,
        )

        if entities_after_cop:
            # Only the FIRST entity after copular gets the hint slot
            first_idx, first_entity = entities_after_cop[0]

            # Check which hint matches the copular subject
            for slot, hints in slot_hints.items():
                for hint in hints:
                    pattern = re.compile(
                        rf"\b{re.escape(hint)}\b",
                        re.IGNORECASE,
                    )
                    hint_match = pattern.search(clause_lower)
                    if hint_match and hint_match.start() < copular_match.start():
                        facts.append(KnowledgeFact(
                            slot=slot,
                            value=first_entity.canonical,
                            confidence=0.85,
                            verb_category="copular",
                            temporal="current",
                            clause=clause,
                        ))
                        matched_entities.add(first_idx)
                        break
                else:
                    continue
                break

    # For entities NOT matched by copular hint, use their own taxonomy slot
    # BUT only if at least one entity WAS matched by copular hint,
    # OR the clause contains a multi-slot trigger word like "stack", "setup"
    # This prevents false positives like "Your goal is to learn Rust" → language: Rust
    multi_slot_triggers = {"stack", "tech stack", "setup", "toolchain",
                           "architecture", "infrastructure", "environment"}
    has_trigger = any(t in clause_lower for t in multi_slot_triggers)

    if matched_entities or has_trigger:
        for i, entity in enumerate(entities):
            if i not in matched_entities:
                facts.append(KnowledgeFact(
                    slot=entity.slot,
                    value=entity.canonical,
                    confidence=0.75,
                    verb_category="copular",
                    temporal="current",
                    clause=clause,
                ))

    return facts


def _infer_migration(
    clause: str,
    verb: RecognizedVerb,
    entities: List[RecognizedEntity],
) -> List[KnowledgeFact]:
    """Handle migration verbs — identify FROM and TO entities.

    "Migrated from MySQL to Postgres" → database: PostgreSQL, deprecated: MySQL
    "Switched to Vue" → just the TO entity
    """
    facts = []
    clause_lower = clause.lower()

    # Determine direction: "from X to Y" or just "to X"
    from_entity: Optional[RecognizedEntity] = None
    to_entity: Optional[RecognizedEntity] = None

    # Check for explicit "from" / "to" keywords
    from_pos = clause_lower.find(" from ")
    to_pos = clause_lower.find(" to ")

    if len(entities) == 1:
        # Single entity — it's the target of migration
        to_entity = entities[0]
    elif len(entities) >= 2:
        # Multiple entities — try to figure out from/to
        # If verb contains "from", first entity after "from" is source
        if "from" in verb.phrase:
            # "migrated from X" — entity near verb is the source
            from_entity = min(entities, key=lambda e: abs(e.start - verb.end))
            to_entity = [e for e in entities if e is not from_entity]
            to_entity = to_entity[0] if to_entity else None
        elif "to" in verb.phrase:
            # "switched to X" — entity near verb is the target
            to_entity = min(entities, key=lambda e: abs(e.start - verb.end))
            from_entity = [e for e in entities if e is not to_entity]
            from_entity = from_entity[0] if from_entity else None
        else:
            # Heuristic: entity after verb is target, entity before is source
            after = [e for e in entities if e.start > verb.end]
            before = [e for e in entities if e.end < verb.start]
            to_entity = after[0] if after else entities[-1]
            from_entity = before[0] if before else None

    if to_entity:
        facts.append(KnowledgeFact(
            slot=to_entity.slot,
            value=to_entity.canonical,
            confidence=verb.confidence,
            verb_category=verb.category,
            temporal="current",
            clause=clause,
            deprecated_value=from_entity.canonical if from_entity else None,
        ))

    if from_entity:
        facts.append(KnowledgeFact(
            slot=from_entity.slot,
            value=from_entity.canonical,
            confidence=verb.confidence,
            verb_category="deprecation",
            temporal="past",
            clause=clause,
        ))

    return facts


def _infer_deprecation(
    verb: RecognizedVerb, entities: List[RecognizedEntity], clause: str = ""
) -> List[KnowledgeFact]:
    """Handle deprecation verbs — mark entities as no longer active."""
    facts = []
    for entity in entities:
        facts.append(KnowledgeFact(
            slot=entity.slot,
            value=entity.canonical,
            confidence=verb.confidence,
            verb_category=verb.category,
            temporal="past",
            clause=clause,
        ))
    return facts


def _infer_standard(
    verb: RecognizedVerb, entities: List[RecognizedEntity]
) -> List[KnowledgeFact]:
    """Handle standard verb categories — adoption, capability, preference, etc."""
    facts = []
    for entity in entities:
        facts.append(KnowledgeFact(
            slot=entity.slot,
            value=entity.canonical,
            confidence=verb.confidence,
            verb_category=verb.category,
            temporal=verb.temporal,
            clause="",
        ))
    return facts


# ── Public API: adapt to GroundCheck ExtractedFact format ─────────────────────

def extract_knowledge_facts(text: str) -> Dict[str, ExtractedFact]:
    """Extract facts using knowledge-based inference.

    Returns dict compatible with GroundCheck's extract_fact_slots() format.
    Only returns facts not already extractable by regex patterns.
    """
    knowledge_facts = infer_facts(text)
    result: Dict[str, ExtractedFact] = {}

    for kf in knowledge_facts:
        # Skip low-confidence tentative facts
        if kf.confidence < 0.40:
            continue

        # Use slot name as key; if multiple entities for same slot,
        # prefer the one with highest confidence
        if kf.slot in result:
            existing = result[kf.slot]
            # If existing has "knowledge" source, compare confidence
            if hasattr(existing, '_knowledge_confidence'):
                if kf.confidence <= existing._knowledge_confidence:
                    continue

        fact = ExtractedFact(
            slot=kf.slot,
            value=kf.value,
            normalized=kf.value.lower(),
            source="knowledge",
        )
        # Stash confidence for comparison (not part of dataclass)
        fact._knowledge_confidence = kf.confidence  # type: ignore[attr-defined]
        result[kf.slot] = fact

    return result


def extract_knowledge_facts_detailed(text: str) -> List[KnowledgeFact]:
    """Extract facts with full metadata (confidence, temporal, verb category).

    Use this when you need the detailed inference results rather than
    the simplified ExtractedFact format.
    """
    return infer_facts(text)
