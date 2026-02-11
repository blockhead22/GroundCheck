"""Tests for knowledge-based fact extraction (Tier 1.5).

These test real conversational sentences that regex extraction misses.
This is the key differentiator for v0.4.0 — understanding natural language
like 'Yeah so we ended up going with Postgres after the MySQL disaster'.
"""

from __future__ import annotations

import pytest

from groundcheck.knowledge_extractor import (
    infer_facts,
    extract_knowledge_facts,
    extract_knowledge_facts_detailed,
    find_entities,
    find_verbs,
    split_into_clauses,
    KnowledgeFact,
    RecognizedEntity,
    RecognizedVerb,
)
from groundcheck import GroundCheck, Memory


# ── Helper ────────────────────────────────────────────────────────────────────

def _fact_dict(text: str) -> dict[str, KnowledgeFact]:
    """Run inference and return dict keyed by (slot, value) for easy lookup."""
    facts = infer_facts(text)
    return {(f.slot, f.value): f for f in facts}


def _slot_values(text: str) -> dict[str, str]:
    """Run extraction and return simple slot → value dict."""
    facts = extract_knowledge_facts(text)
    return {k: v.value for k, v in facts.items()}


# ── Clause splitting ─────────────────────────────────────────────────────────

class TestClauseSplitting:
    """Verify text gets split into meaningful clauses."""

    def test_period_boundary(self):
        clauses = split_into_clauses("We use Postgres. Redis for caching.")
        assert len(clauses) == 2

    def test_semicolon_boundary(self):
        clauses = split_into_clauses("Backend is FastAPI; frontend is React")
        assert len(clauses) == 2

    def test_conjunction_boundary(self):
        clauses = split_into_clauses("We chose Postgres but dropped MySQL")
        assert len(clauses) >= 2

    def test_comma_boundary(self):
        clauses = split_into_clauses("Dropped Redis, switched to Memcached")
        assert len(clauses) == 2

    def test_after_boundary(self):
        clauses = split_into_clauses("went with Postgres after the MySQL disaster")
        assert len(clauses) >= 2

    def test_short_fragments_removed(self):
        clauses = split_into_clauses("We use Postgres. OK.")
        # "OK." is only 2 chars, should be dropped
        assert all(len(c) >= 5 for c in clauses)

    def test_single_clause_preserved(self):
        clauses = split_into_clauses("We use PostgreSQL")
        assert clauses == ["We use PostgreSQL"]


# ── Entity recognition ───────────────────────────────────────────────────────

class TestEntityRecognition:
    """Verify taxonomy-based entity detection."""

    def test_finds_database(self):
        ents = find_entities("We chose PostgreSQL for our database")
        names = [e.canonical for e in ents]
        assert "PostgreSQL" in names

    def test_finds_multiple_entities(self):
        ents = find_entities("FastAPI with React and PostgreSQL on AWS")
        names = {e.canonical for e in ents}
        assert names >= {"FastAPI", "React", "PostgreSQL", "AWS"}

    def test_word_boundary_respected(self):
        # "Rustacean" should NOT match "Rust"
        ents = find_entities("I am a proud Rustacean")
        assert len(ents) == 0

    def test_longer_entity_preferred(self):
        # "GitHub Actions" should match as one entity, not "GitHub" alone
        ents = find_entities("We use GitHub Actions for CI")
        names = [e.canonical for e in ents]
        assert "GitHub Actions" in names

    def test_correct_slot_assigned(self):
        ents = find_entities("PostgreSQL")
        assert ents[0].slot == "database"

        ents = find_entities("React")
        assert ents[0].slot == "frontend_framework"

        ents = find_entities("AWS")
        assert ents[0].slot == "cloud_provider"

    def test_case_insensitive(self):
        ents = find_entities("we use postgresql and fastapi")
        names = {e.canonical for e in ents}
        assert "PostgreSQL" in names
        assert "FastAPI" in names


# ── Verb recognition ─────────────────────────────────────────────────────────

class TestVerbRecognition:
    """Verify ontology-based verb detection."""

    def test_simple_adoption_verb(self):
        verbs = find_verbs("We use PostgreSQL")
        assert any(v.category == "adoption" for v in verbs)

    def test_migration_verb(self):
        verbs = find_verbs("We migrated to PostgreSQL")
        assert any(v.category == "migration" for v in verbs)

    def test_deprecation_verb(self):
        verbs = find_verbs("We dropped MySQL")
        assert any(v.category == "deprecation" for v in verbs)

    def test_tentative_verb(self):
        verbs = find_verbs("We're considering Vue")
        assert any(v.category == "tentative" for v in verbs)

    def test_multi_word_verb_preferred(self):
        # "switched to" should match as one verb, not "switched" alone
        verbs = find_verbs("We switched to PostgreSQL")
        assert any(v.phrase == "switched to" for v in verbs)

    def test_preference_verb(self):
        verbs = find_verbs("I love FastAPI")
        assert any(v.category == "preference" for v in verbs)


# ── Adoption extraction ──────────────────────────────────────────────────────

class TestAdoptionExtraction:
    """Test extraction of adoption facts from conversational text."""

    def test_simple_use(self):
        facts = _fact_dict("We use PostgreSQL")
        assert ("database", "PostgreSQL") in facts
        f = facts[("database", "PostgreSQL")]
        assert f.verb_category == "adoption"
        assert f.confidence >= 0.85

    def test_informal_adoption(self):
        facts = _fact_dict("Yeah so we ended up going with Postgres")
        assert ("database", "Postgres") in facts
        f = facts[("database", "Postgres")]
        assert f.verb_category == "adoption"

    def test_deploy_on(self):
        facts = _fact_dict("We deploy on AWS")
        assert ("cloud_provider", "AWS") in facts

    def test_built_with(self):
        facts = _fact_dict("The backend is built with FastAPI")
        assert ("backend_framework", "FastAPI") in facts

    def test_no_false_positive_go_language(self):
        """'go with' should NOT extract 'Go' as a language."""
        facts = _fact_dict("Had to go with MongoDB")
        # MongoDB should be found, Go should NOT
        assert ("database", "MongoDB") in facts
        assert ("language", "Go") not in facts


# ── Migration extraction ─────────────────────────────────────────────────────

class TestMigrationExtraction:
    """Test migration verb handling with FROM/TO entity detection."""

    def test_explicit_from_to(self):
        facts = _fact_dict("We migrated from MySQL to PostgreSQL")
        f = facts[("database", "PostgreSQL")]
        assert f.verb_category == "migration"
        assert f.deprecated_value == "MySQL"
        assert f.temporal == "current"

    def test_implicit_migration_with_to(self):
        facts = _fact_dict("Switched to Memcached")
        assert ("database", "Memcached") in facts
        f = facts[("database", "Memcached")]
        assert f.verb_category == "migration"

    def test_went_back_to(self):
        facts = _fact_dict("went back to Node.js")
        assert ("language", "Node.js") in facts
        f = facts[("language", "Node.js")]
        assert f.verb_category == "migration"


# ── Deprecation extraction ───────────────────────────────────────────────────

class TestDeprecationExtraction:
    """Test deprecation verb handling."""

    def test_simple_deprecation(self):
        facts = _fact_dict("We dropped MySQL")
        assert ("database", "MySQL") in facts
        f = facts[("database", "MySQL")]
        assert f.verb_category == "deprecation"
        assert f.temporal == "past"

    def test_ripped_out(self):
        facts = _fact_dict("Ripped out Jenkins")
        assert ("ci_cd", "Jenkins") in facts

    def test_no_more_x(self):
        facts = _fact_dict("no more JavaScript")
        assert ("language", "JavaScript") in facts
        f = facts[("language", "JavaScript")]
        assert f.verb_category == "deprecation"

    def test_negative_context_without_verb(self):
        """'MySQL disaster' should infer deprecation from negative context."""
        facts = _fact_dict("the whole MySQL disaster")
        assert ("database", "MySQL") in facts
        f = facts[("database", "MySQL")]
        assert f.verb_category == "deprecation"
        assert f.temporal == "past"


# ── Tentative extraction ─────────────────────────────────────────────────────

class TestTentativeExtraction:
    """Test tentative/uncertain fact extraction."""

    def test_considering(self):
        facts = _fact_dict("considering Vue")
        assert ("frontend_framework", "Vue") in facts
        f = facts[("frontend_framework", "Vue")]
        assert f.verb_category == "tentative"
        assert f.confidence < 0.50

    def test_evaluating(self):
        facts = _fact_dict("Currently evaluating Vue")
        assert ("frontend_framework", "Vue") in facts
        f = facts[("frontend_framework", "Vue")]
        assert f.confidence < 0.50

    def test_tentative_overrides_migration(self):
        """'Considering switching to X' should be tentative, not migration."""
        facts = _fact_dict("Considering switching to Rust")
        assert ("language", "Rust") in facts
        f = facts[("language", "Rust")]
        assert f.verb_category == "tentative"
        assert f.confidence < 0.50

    def test_tried(self):
        facts = _fact_dict("Tried Deno")
        assert ("language", "Deno") in facts
        f = facts[("language", "Deno")]
        assert f.verb_category == "tentative"


# ── Skip patterns ────────────────────────────────────────────────────────────

class TestSkipPatterns:
    """Test that questions, hedges, and uncertainty are skipped."""

    def test_question_skipped(self):
        facts = infer_facts("What database are we using?")
        assert len(facts) == 0

    def test_not_sure_skipped(self):
        facts = infer_facts("Not sure if we should go with Svelte or Vue")
        assert len(facts) == 0

    def test_uncertainty_skipped(self):
        facts = infer_facts("I don't know if we use Redis")
        assert len(facts) == 0


# ── Copular inference ────────────────────────────────────────────────────────

class TestCopularInference:
    """Test 'our X is Y' pattern extraction."""

    def test_backend_is_fastapi(self):
        facts = _fact_dict("Our backend is FastAPI")
        assert ("backend_framework", "FastAPI") in facts
        f = facts[("backend_framework", "FastAPI")]
        assert f.verb_category == "copular"

    def test_backend_with_frontend_different_slots(self):
        """Backend copular should NOT force React into backend_framework."""
        facts = _fact_dict("Our backend is FastAPI with a React frontend")
        assert ("backend_framework", "FastAPI") in facts
        assert ("frontend_framework", "React") in facts

    def test_taxonomy_fallback_for_unmatched_entities(self):
        """Entities in multi-slot contexts use their taxonomy slot."""
        sv = _slot_values("Our stack is Python, FastAPI, PostgreSQL, and Redis on AWS")
        # "stack" triggers multi-slot expansion
        assert len(sv) >= 3


# ── Verb context inheritance ─────────────────────────────────────────────────

class TestVerbInheritance:
    """Test that verbless clauses inherit verb context from previous clause."""

    def test_and_list_inheritance(self):
        """'We use X for CI and Y for monitoring' — Y should get adoption too."""
        sv = _slot_values("We use GitHub Actions for CI and Datadog for monitoring")
        assert "ci_cd" in sv
        assert sv["ci_cd"] == "GitHub Actions"
        assert "monitoring" in sv
        assert sv["monitoring"] == "Datadog"


# ── Deduplication ────────────────────────────────────────────────────────────

class TestDeduplication:
    """Test that migration facts suppress redundant deprecation/adoption facts."""

    def test_migration_suppresses_extra_deprecation(self):
        """'Dropped Redis, switched to Memcached' should not double-count."""
        facts = infer_facts("Dropped Redis, switched to Memcached")
        # Redis should appear once (deprecation), Memcached once (migration)
        redis_facts = [f for f in facts if f.value == "Redis"]
        memcached_facts = [f for f in facts if f.value == "Memcached"]
        assert len(redis_facts) == 1
        assert redis_facts[0].verb_category == "deprecation"
        assert len(memcached_facts) == 1
        assert memcached_facts[0].verb_category == "migration"


# ── Position-aware matching ──────────────────────────────────────────────────

class TestPositionAwareMatching:
    """Test that entities match their nearest verb when multiple verbs exist."""

    def test_deprecation_plus_preference(self):
        """'Ripped out Jenkins, way happier with GitHub Actions'
        — Jenkins→deprecation, GitHub Actions→preference."""
        facts = _fact_dict("Ripped out Jenkins last month, way happier with GitHub Actions")
        j = facts.get(("ci_cd", "Jenkins"))
        gh = facts.get(("ci_cd", "GitHub Actions"))
        assert j is not None
        assert j.verb_category == "deprecation"
        assert gh is not None
        assert gh.verb_category == "preference"


# ── Complex real-world sentences ─────────────────────────────────────────────

class TestRealWorldSentences:
    """The core test suite — real conversational text that regex would miss."""

    def test_postgres_mysql_disaster(self):
        sv = _slot_values("Yeah so we ended up going with Postgres after the whole MySQL disaster")
        assert sv.get("database") == "Postgres"

    def test_migration_from_to(self):
        sv = _slot_values("We migrated from MySQL to PostgreSQL last quarter")
        assert sv.get("database") == "PostgreSQL"

    def test_typescript_no_more_javascript(self):
        facts = infer_facts("TypeScript everywhere, no more JavaScript")
        ts = [f for f in facts if f.value == "TypeScript"]
        js = [f for f in facts if f.value == "JavaScript"]
        # TypeScript should be current, JavaScript deprecated
        assert any(f.temporal == "current" for f in ts)
        assert any(f.verb_category == "deprecation" for f in js)

    def test_full_stack_description(self):
        sv = _slot_values("Our stack is Python, FastAPI, PostgreSQL, and Redis on AWS")
        # Should find at least 3 of the 5 entities
        assert len(sv) >= 3

    def test_evaluating_vs_current(self):
        facts = infer_facts("Currently evaluating Vue but still on React")
        vue = [f for f in facts if f.value == "Vue"]
        react = [f for f in facts if f.value == "React"]
        assert any(f.verb_category == "tentative" for f in vue)
        assert len(react) > 0  # React should be recognized

    def test_kafka_replaced_rabbitmq(self):
        facts = infer_facts("Kafka handles all our event streaming, replaced RabbitMQ")
        kafka = [f for f in facts if f.value == "Kafka"]
        rabbit = [f for f in facts if f.value == "RabbitMQ"]
        assert any(f.verb_category == "capability" for f in kafka)
        assert any(f.verb_category == "deprecation" for f in rabbit)

    def test_thinking_about_replacing(self):
        facts = infer_facts("Thinking about replacing Heroku with Fly.io")
        flyio = [f for f in facts if f.value == "Fly.io"]
        assert any(f.verb_category == "tentative" for f in flyio)


# ── Integration with verifier (extract_claims) ──────────────────────────────

class TestVerifierIntegration:
    """Test that knowledge extraction supplements regex in the verifier."""

    def test_extract_claims_finds_conversational_facts(self):
        gc = GroundCheck()
        claims = gc.extract_claims("We migrated from MySQL to PostgreSQL")
        assert "database" in claims
        assert claims["database"].value == "PostgreSQL"

    def test_extract_claims_finds_adoption(self):
        gc = GroundCheck()
        claims = gc.extract_claims("We use GitHub Actions for CI and Datadog for monitoring")
        assert "ci_cd" in claims
        assert "monitoring" in claims

    def test_verify_catches_contradiction_via_knowledge(self):
        """Knowledge extraction should detect contradictions regex would miss."""
        gc = GroundCheck()
        memories = [Memory(id="m1", text="FACT: database = MySQL", trust=0.9)]
        result = gc.verify(
            "We ended up going with Postgres instead",
            memories,
        )
        # Should fail because Postgres contradicts stored MySQL
        assert result.passed is False

    def test_verify_passes_when_aligned(self):
        gc = GroundCheck()
        memories = [Memory(id="m1", text="FACT: database = PostgreSQL", trust=0.9)]
        result = gc.verify(
            "We migrated to PostgreSQL last quarter",
            memories,
        )
        assert result.passed is True

    def test_knowledge_does_not_override_regex(self):
        """Regex-extracted facts should not be overwritten by knowledge facts."""
        gc = GroundCheck()
        claims = gc.extract_claims("User works at Microsoft and uses PostgreSQL")
        # "employer" from regex, "database" from knowledge
        # Both should be present
        if "employer" in claims:
            assert claims["employer"].source != "knowledge"


# ── extract_knowledge_facts API format ───────────────────────────────────────

class TestExtractKnowledgeFactsAPI:
    """Test the public API returns the correct format."""

    def test_returns_dict_of_extracted_facts(self):
        result = extract_knowledge_facts("We use PostgreSQL")
        assert isinstance(result, dict)
        assert "database" in result
        fact = result["database"]
        assert fact.slot == "database"
        assert fact.value == "PostgreSQL"
        assert fact.source == "knowledge"

    def test_detailed_returns_knowledge_facts(self):
        result = extract_knowledge_facts_detailed("We use PostgreSQL")
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], KnowledgeFact)

    def test_low_confidence_filtered(self):
        """Tentative facts below threshold should be filtered from simple API."""
        result = extract_knowledge_facts("Maybe considering Vue someday")
        # Tentative with conf=0.35 should be filtered (threshold is 0.40)
        assert "frontend_framework" not in result


# ── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_string(self):
        facts = infer_facts("")
        assert facts == []

    def test_no_entities(self):
        facts = infer_facts("The weather is nice today")
        assert facts == []

    def test_very_long_input(self):
        text = "We use PostgreSQL. " * 100
        facts = infer_facts(text)
        # Should still work without crashing
        assert len(facts) > 0

    def test_mixed_case_entities(self):
        facts = _fact_dict("we USE postgresql AND fastapi")
        assert ("database", "PostgreSQL") in facts
        assert ("backend_framework", "FastAPI") in facts

    def test_unicode_resilience(self):
        facts = infer_facts("We use PostgreSQL 🚀 for everything")
        assert len(facts) > 0

    def test_multiple_sentences(self):
        text = "We use PostgreSQL for data. React for the frontend. AWS for hosting."
        facts = infer_facts(text)
        slots = {f.slot for f in facts}
        assert "database" in slots
        assert "frontend_framework" in slots
        assert "cloud_provider" in slots
