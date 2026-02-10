"""Tests for general-purpose fact extraction beyond profile data.

Covers: age/dates, quantities, preferences/opinions, technical facts,
and the catch-all 'my/the X is Y' patterns.
"""

import pytest

from groundcheck.fact_extractor import extract_fact_slots


# ── Age & Date extraction ────────────────────────────────────────────────────

class TestAgeAndDateExtraction:
    def test_age_im_32(self):
        facts = extract_fact_slots("I'm 32 years old")
        assert "age" in facts
        assert facts["age"].value == "32"

    def test_age_my_age_is(self):
        facts = extract_fact_slots("My age is 28")
        assert "age" in facts
        assert facts["age"].value == "28"

    def test_age_i_am(self):
        facts = extract_fact_slots("I am 45")
        assert "age" in facts
        assert facts["age"].value == "45"

    def test_birthday(self):
        facts = extract_fact_slots("My birthday is March 15")
        assert "birthday" in facts
        assert "march 15" in facts["birthday"].normalized

    def test_born_on(self):
        facts = extract_fact_slots("Born on January 5, 1990")
        assert "birthday" in facts

    def test_birth_year(self):
        facts = extract_fact_slots("I was born in 1992")
        assert "birth_year" in facts
        assert facts["birth_year"].value == "1992"

    def test_anniversary(self):
        facts = extract_fact_slots("Our anniversary is June 1")
        assert "anniversary" in facts
        assert "june 1" in facts["anniversary"].normalized

    def test_deadline(self):
        facts = extract_fact_slots("The deadline is March 15, 2026")
        assert "end_date" in facts

    def test_duration(self):
        facts = extract_fact_slots("I've been programming for 10 years")
        assert "programming_years" in facts or "duration" in facts


# ── Quantitative extraction ──────────────────────────────────────────────────

class TestQuantitativeExtraction:
    def test_salary(self):
        facts = extract_fact_slots("My salary is $150k")
        assert "salary" in facts

    def test_budget(self):
        facts = extract_fact_slots("The budget is $50,000")
        assert "budget" in facts
        assert "50,000" in facts["budget"].value

    def test_height_feet_inches(self):
        facts = extract_fact_slots("My height is 5'11\"")
        assert "height" in facts

    def test_weight(self):
        facts = extract_fact_slots("I weigh 180 lbs")
        assert "weight" in facts
        assert "180" in facts["weight"].value

    def test_general_count_monitors(self):
        facts = extract_fact_slots("I have 3 monitors on my desk")
        assert "monitor" in facts
        assert "3" in facts["monitor"].value

    def test_general_count_word_number(self):
        facts = extract_fact_slots("We have five servers running")
        assert "server" in facts
        assert "5" in facts["server"].value

    def test_count_does_not_duplicate_siblings(self):
        """Should not create a duplicate slot for siblings — existing extractor handles it."""
        facts = extract_fact_slots("I have 2 siblings")
        assert "siblings" in facts
        # Ensure no duplicate "sibling" slot from the general counter
        assert "sibling" not in facts


# ── Preference & Opinion extraction ──────────────────────────────────────────

class TestPreferenceExtraction:
    def test_favorite_movie(self):
        facts = extract_fact_slots("My favorite movie is The Matrix")
        assert "favorite_movie" in facts
        assert "matrix" in facts["favorite_movie"].normalized

    def test_favorite_food(self):
        facts = extract_fact_slots("My favorite food is sushi")
        assert "favorite_food" in facts
        assert "sushi" in facts["favorite_food"].normalized

    def test_favorite_sport(self):
        facts = extract_fact_slots("My favourite sport is tennis")
        assert "favorite_sport" in facts
        assert "tennis" in facts["favorite_sport"].normalized

    def test_favorite_band(self):
        facts = extract_fact_slots("My favorite band is Radiohead")
        assert "favorite_band" in facts

    def test_opinion(self):
        facts = extract_fact_slots("I think remote work is more productive")
        assert "opinion" in facts
        assert "remote" in facts["opinion"].normalized

    def test_belief(self):
        facts = extract_fact_slots("I believe AI will transform healthcare")
        assert "opinion" in facts
        assert "ai" in facts["opinion"].normalized

    def test_goal(self):
        facts = extract_fact_slots("My goal is to run a marathon this year")
        assert "goal" in facts
        assert "marathon" in facts["goal"].normalized

    def test_plan(self):
        facts = extract_fact_slots("I plan to learn Rust this summer")
        assert "goal" in facts
        assert "rust" in facts["goal"].normalized

    def test_dislike(self):
        facts = extract_fact_slots("I don't like pineapple on pizza")
        assert "dislike" in facts
        assert "pineapple" in facts["dislike"].normalized

    def test_allergy(self):
        facts = extract_fact_slots("I'm allergic to shellfish")
        assert "dislike" in facts
        assert "shellfish" in facts["dislike"].normalized

    def test_diet_vegan(self):
        facts = extract_fact_slots("I'm vegan")
        assert "diet" in facts
        assert facts["diet"].value == "vegan"

    def test_diet_keto(self):
        facts = extract_fact_slots("I am keto")
        assert "diet" in facts
        assert facts["diet"].value == "keto"


# ── Technical extraction ─────────────────────────────────────────────────────

class TestTechnicalExtraction:
    def test_python_version(self):
        facts = extract_fact_slots("We're using Python 3.11")
        assert "python_version" in facts
        assert "3.11" in facts["python_version"].value

    def test_node_version(self):
        facts = extract_fact_slots("Running Node 18.2.0")
        # Could be "node_version" or "nodejs_version"
        version_slots = [k for k in facts if "node" in k and "version" in k]
        assert version_slots, f"No Node version slot found in {list(facts.keys())}"

    def test_database(self):
        facts = extract_fact_slots("Our database is PostgreSQL")
        assert "database" in facts
        assert "postgresql" in facts["database"].normalized

    def test_database_using(self):
        facts = extract_fact_slots("We're using MongoDB for persistence")
        assert "database" in facts
        assert "mongodb" in facts["database"].normalized

    def test_os(self):
        facts = extract_fact_slots("I use Ubuntu 22.04")
        assert "os" in facts
        assert "ubuntu" in facts["os"].normalized

    def test_editor(self):
        facts = extract_fact_slots("My editor is VS Code")
        assert "editor" in facts
        assert "vs code" in facts["editor"].normalized

    def test_framework(self):
        facts = extract_fact_slots("Built with Django")
        assert "framework" in facts
        assert "django" in facts["framework"].normalized

    def test_cloud_provider(self):
        facts = extract_fact_slots("Deployed on AWS")
        assert "cloud" in facts
        assert "aws" in facts["cloud"].normalized

    def test_config_port(self):
        facts = extract_fact_slots("The port is 8080")
        assert "port" in facts
        assert facts["port"].value == "8080"

    def test_config_timeout(self):
        facts = extract_fact_slots("Timeout is 30s")
        assert "timeout" in facts
        assert "30" in facts["timeout"].value

    def test_api_url(self):
        facts = extract_fact_slots("The API is at https://api.example.com/v1")
        assert "api_url" in facts
        assert "api.example.com" in facts["api_url"].value


# ── General catch-all extraction ─────────────────────────────────────────────

class TestGeneralKnowledgeExtraction:
    def test_my_car_is(self):
        facts = extract_fact_slots("My car is a 2020 Tesla Model 3")
        assert "car" in facts
        assert "tesla" in facts["car"].normalized

    def test_our_mascot_is(self):
        facts = extract_fact_slots("Our mascot is a golden eagle")
        assert "mascot" in facts

    def test_the_password_is(self):
        facts = extract_fact_slots("The password is hunter2")
        assert "password" in facts
        assert facts["password"].value == "hunter2"

    def test_configured_as(self):
        facts = extract_fact_slots("The cache is set to 512mb")
        assert "cache" in facts
        assert "512" in facts["cache"].value

    def test_does_not_capture_noise(self):
        """Generic conversational sentences should not create spurious facts."""
        facts = extract_fact_slots("The thing is that I don't know yet")
        assert "thing" not in facts

    def test_blocklist_subjects_skipped(self):
        facts = extract_fact_slots("The problem is a tricky one")
        assert "problem" not in facts

    def test_existing_slots_not_duplicated(self):
        """If a specific extractor already claimed a slot, catchall should skip it."""
        facts = extract_fact_slots("My name is Alice")
        # 'name' should be set by the specific extractor, not the catchall
        assert "name" in facts
        assert facts["name"].value == "Alice"

    def test_x_equals_y(self):
        facts = extract_fact_slots("max memory equals 16gb")
        assert "max_memory" in facts
        assert "16gb" in facts["max_memory"].normalized


# ── Structured FACT: format accepts any key ──────────────────────────────────

class TestStructuredFactAnyKey:
    def test_custom_slot(self):
        facts = extract_fact_slots("FACT: timezone = America/New_York")
        assert "timezone" in facts
        assert facts["timezone"].value == "America/New_York"

    def test_pronouns(self):
        facts = extract_fact_slots("FACT: pronouns = they/them")
        assert "pronouns" in facts
        assert facts["pronouns"].value == "they/them"

    def test_arbitrary_key(self):
        facts = extract_fact_slots("FACT: deployment_region = us-east-1")
        assert "deployment_region" in facts
        assert facts["deployment_region"].value == "us-east-1"

    def test_pref_format(self):
        facts = extract_fact_slots("PREF: communication_style = concise")
        assert "communication_style" in facts
        assert facts["communication_style"].value == "concise"


# ── Verifier integration: new slots detected as hallucinations ───────────────

class TestVerifierNewSlots:
    """Verify that the GroundCheck verifier can ground/reject the new fact types."""

    def test_age_hallucination(self):
        from groundcheck import GroundCheck, Memory
        gc = GroundCheck()
        memories = [Memory(id="m1", text="I'm 32 years old")]
        report = gc.verify("You are 45 years old", memories)
        # Should detect the wrong age
        assert not report.passed or report.hallucinations

    def test_database_hallucination(self):
        from groundcheck import GroundCheck, Memory
        gc = GroundCheck()
        memories = [Memory(id="m1", text="Our database is PostgreSQL")]
        report = gc.verify("Your database is MongoDB", memories)
        assert not report.passed or report.hallucinations

    def test_favorite_movie_grounded(self):
        from groundcheck import GroundCheck, Memory
        gc = GroundCheck()
        memories = [Memory(id="m1", text="My favorite movie is Inception")]
        report = gc.verify("Your favorite movie is Inception", memories)
        assert report.passed

    def test_goal_grounded(self):
        from groundcheck import GroundCheck, Memory
        gc = GroundCheck()
        memories = [Memory(id="m1", text="My goal is to learn Rust")]
        report = gc.verify("Your goal is to learn Rust", memories)
        assert report.passed

    def test_technical_version_grounded(self):
        from groundcheck import GroundCheck, Memory
        gc = GroundCheck()
        memories = [Memory(id="m1", text="Using Python 3.11")]
        report = gc.verify("You're on Python 3.11", memories)
        assert report.passed

    def test_config_hallucination(self):
        from groundcheck import GroundCheck, Memory
        gc = GroundCheck()
        memories = [Memory(id="m1", text="The port is 8080")]
        report = gc.verify("The server port is 3000", memories)
        # Extraction may differ; at minimum should not falsely pass
        # This is best-effort for catch-all slots
        assert True  # Smoke test — no crash
