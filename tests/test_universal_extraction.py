"""Tests for universal fact extraction — ensuring the catch-all extractor
handles real-world declarative claims beyond personal profile slots.

These 10 sentences were identified by the extraction audit as the critical
gap: 6 of 10 were DROPPED by the old extractor, 2 were partial.
"""

import pytest
from groundcheck.fact_extractor import extract_fact_slots
from groundcheck import GroundCheck, Memory


# ─── Universal Extraction: 10 Audit Sentences ───────────────────────────────

class TestUniversalExtraction:
    """Every sentence from the external audit must produce at least one fact."""

    def test_api_rate_limit(self):
        """'The API rate limit is 1000 requests per minute'"""
        facts = extract_fact_slots("The API rate limit is 1000 requests per minute.")
        # Should extract something like api_rate_limit -> 1000 requests per minute
        values = " ".join(f.value for f in facts.values()).lower()
        assert "1000" in values, f"Expected '1000' in extracted values, got: {facts}"

    def test_rest_not_graphql(self):
        """'We agreed to use REST not GraphQL'"""
        facts = extract_fact_slots("We agreed to use REST not GraphQL.")
        assert len(facts) >= 1, f"Expected at least 1 fact, got: {facts}"
        values = " ".join(f.value for f in facts.values()).lower()
        assert "rest" in values, f"Expected 'REST' in values, got: {facts}"

    def test_hipaa_compliance(self):
        """'The client requires HIPAA compliance'"""
        facts = extract_fact_slots("The client requires HIPAA compliance.")
        assert len(facts) >= 1, f"Expected at least 1 fact, got: {facts}"
        values = " ".join(f.value for f in facts.values()).lower()
        assert "hipaa" in values, f"Expected 'HIPAA' in values, got: {facts}"

    def test_max_retries(self):
        """'Max retries should be 5'"""
        facts = extract_fact_slots("Max retries should be 5.")
        assert len(facts) >= 1, f"Expected at least 1 fact, got: {facts}"
        values = " ".join(f.value for f in facts.values()).lower()
        assert "5" in values, f"Expected '5' in values, got: {facts}"

    def test_microservices_architecture(self):
        """'The project uses microservices architecture'"""
        facts = extract_fact_slots("The project uses microservices architecture.")
        assert len(facts) >= 1, f"Expected at least 1 fact, got: {facts}"
        values = " ".join(f.value for f in facts.values()).lower()
        assert "microservice" in values, f"Expected 'microservice' in values, got: {facts}"

    def test_postgresql_and_redis(self):
        """'We need to support PostgreSQL and Redis'"""
        facts = extract_fact_slots("We need to support PostgreSQL and Redis.")
        assert len(facts) >= 1, f"Expected at least 1 fact, got: {facts}"
        values = " ".join(f.value for f in facts.values()).lower()
        assert "postgresql" in values or "redis" in values, f"Got: {facts}"

    def test_sla_uptime(self):
        """'Our SLA requires 99.9% uptime'"""
        facts = extract_fact_slots("Our SLA requires 99.9% uptime.")
        assert len(facts) >= 1, f"Expected at least 1 fact, got: {facts}"
        values = " ".join(f.value for f in facts.values()).lower()
        assert "99.9" in values or "uptime" in values, f"Got: {facts}"

    def test_react_and_fastapi(self):
        """'The frontend is React, backend is FastAPI' — should get BOTH facts."""
        facts = extract_fact_slots("The frontend is React, backend is FastAPI.")
        assert len(facts) >= 2, f"Expected at least 2 facts, got: {facts}"
        values = " ".join(f.value for f in facts.values()).lower()
        assert "react" in values, f"Expected 'React' in values, got: {facts}"
        assert "fastapi" in values, f"Expected 'FastAPI' in values, got: {facts}"

    def test_sprint_deadline(self):
        """'The sprint deadline is Friday'"""
        facts = extract_fact_slots("The sprint deadline is Friday.")
        assert len(facts) >= 1, f"Expected at least 1 fact, got: {facts}"
        values = " ".join(f.value for f in facts.values()).lower()
        assert "friday" in values, f"Expected 'Friday' in values, got: {facts}"

    def test_deployment_target(self):
        """'The deployment target is Kubernetes on AWS'"""
        facts = extract_fact_slots("The deployment target is Kubernetes on AWS.")
        assert len(facts) >= 1, f"Expected at least 1 fact, got: {facts}"
        values = " ".join(f.value for f in facts.values()).lower()
        assert "kubernetes" in values or "aws" in values, f"Got: {facts}"


# ─── Additional Universal Patterns ──────────────────────────────────────────

class TestNonCopularVerbs:
    """Test extraction of subject-verb-object patterns beyond is/are."""

    def test_project_uses(self):
        facts = extract_fact_slots("The project uses Docker for containerization.")
        values = " ".join(f.value for f in facts.values()).lower()
        assert "docker" in values, f"Got: {facts}"

    def test_system_handles(self):
        facts = extract_fact_slots("The system handles authentication via OAuth2.")
        values = " ".join(f.value for f in facts.values()).lower()
        assert "oauth2" in values or "authentication" in values, f"Got: {facts}"

    def test_backend_runs(self):
        facts = extract_fact_slots("Our backend runs on Python 3.11.")
        values = " ".join(f.value for f in facts.values()).lower()
        assert "python" in values, f"Got: {facts}"

    def test_app_supports(self):
        facts = extract_fact_slots("The app supports multi-tenancy.")
        values = " ".join(f.value for f in facts.values()).lower()
        assert "multi-tenancy" in values or "tenancy" in values, f"Got: {facts}"


class TestDecisionPatterns:
    """Test extraction of agreement/decision claims."""

    def test_team_decided(self):
        facts = extract_fact_slots("We decided to use PostgreSQL.")
        assert len(facts) >= 1, f"Expected at least 1 fact, got: {facts}"
        values = " ".join(f.value for f in facts.values()).lower()
        assert "postgresql" in values, f"Got: {facts}"

    def test_chose_architecture(self):
        facts = extract_fact_slots("The team chose microservices over monolith.")
        assert len(facts) >= 1, f"Expected at least 1 fact, got: {facts}"
        values = " ".join(f.value for f in facts.values()).lower()
        assert "microservice" in values, f"Got: {facts}"


class TestPrescriptivePatterns:
    """Test 'should be / must be' patterns."""

    def test_timeout_should_be(self):
        facts = extract_fact_slots("The timeout should be 30 seconds.")
        assert len(facts) >= 1, f"Expected at least 1 fact, got: {facts}"
        values = " ".join(f.value for f in facts.values()).lower()
        assert "30" in values, f"Got: {facts}"

    def test_password_must_be(self):
        facts = extract_fact_slots("Password length must be at least 12 characters.")
        assert len(facts) >= 1, f"Expected at least 1 fact, got: {facts}"
        values = " ".join(f.value for f in facts.values()).lower()
        assert "12" in values, f"Got: {facts}"

    def test_response_needs_to_be(self):
        facts = extract_fact_slots("Response time needs to be under 200ms.")
        assert len(facts) >= 1, f"Expected at least 1 fact, got: {facts}"
        values = " ".join(f.value for f in facts.values()).lower()
        assert "200" in values, f"Got: {facts}"


class TestClauseSplitting:
    """Test that comma/semicolon-separated clauses each produce facts."""

    def test_semicolon_split(self):
        facts = extract_fact_slots("The database is PostgreSQL; the cache is Redis.")
        assert len(facts) >= 2, f"Expected 2+ facts from clause split, got: {facts}"

    def test_comma_split(self):
        facts = extract_fact_slots("The frontend is React, the backend is Django.")
        assert len(facts) >= 2, f"Expected 2+ facts from comma split, got: {facts}"


# ─── Dynamic Contradiction Detection ────────────────────────────────────────

class TestDynamicContradictions:
    """Verify that dynamically-extracted slots (not in KNOWN_EXCLUSIVE_SLOTS)
    still get flagged as contradictions when values differ."""

    def test_dynamic_slot_contradiction(self):
        """Two memories with different values for a dynamic slot should contradict."""
        gc = GroundCheck()
        memories = [
            Memory(id="m1", text="The sprint deadline is Friday.", trust=0.9),
            Memory(id="m2", text="The sprint deadline is Monday.", trust=0.9),
        ]
        result = gc.verify("The sprint deadline is Friday.", memories)
        # Should detect contradiction on sprint_deadline slot
        assert len(result.contradiction_details) >= 1, (
            f"Expected contradiction for sprint_deadline, got: {result.contradiction_details}"
        )

    def test_dynamic_slot_no_false_positive(self):
        """Two memories with the SAME value for a dynamic slot should NOT contradict."""
        gc = GroundCheck()
        memories = [
            Memory(id="m1", text="The sprint deadline is Friday.", trust=0.9),
            Memory(id="m2", text="Our sprint deadline is Friday.", trust=0.9),
        ]
        result = gc.verify("The sprint deadline is Friday.", memories)
        # sprint_deadline should not show up as contradiction
        deadline_contradictions = [c for c in result.contradiction_details if "deadline" in c.slot]
        assert len(deadline_contradictions) == 0, (
            f"False positive contradiction: {deadline_contradictions}"
        )

    def test_known_slot_still_works(self):
        """Known-exclusive slots like 'employer' should still detect contradictions."""
        gc = GroundCheck()
        memories = [
            Memory(id="m1", text="I work at Google.", trust=0.9),
            Memory(id="m2", text="I work at Amazon.", trust=0.9),
        ]
        result = gc.verify("I work at Google.", memories)
        assert len(result.contradiction_details) >= 1

    def test_additive_slot_no_contradiction(self):
        """Additive slots (like skill) should not trigger contradictions with unique values."""
        gc = GroundCheck()
        memories = [
            Memory(id="m1", text="My skill is Python.", trust=0.9),
            Memory(id="m2", text="My skill is JavaScript.", trust=0.9),
        ]
        result = gc.verify("My skill is Python.", memories)
        skill_contradictions = [c for c in result.contradiction_details if c.slot == "skill"]
        # skill is in ADDITIVE_SLOTS, so should NOT trigger
        assert len(skill_contradictions) == 0, (
            f"Skill should be additive, got contradiction: {skill_contradictions}"
        )


# ─── Backward Compatibility ─────────────────────────────────────────────────

class TestBackwardCompatibility:
    """Ensure the alias MUTUALLY_EXCLUSIVE_SLOTS still works."""

    def test_alias_exists(self):
        gc = GroundCheck()
        assert hasattr(gc, 'MUTUALLY_EXCLUSIVE_SLOTS')
        assert gc.MUTUALLY_EXCLUSIVE_SLOTS is gc.KNOWN_EXCLUSIVE_SLOTS
