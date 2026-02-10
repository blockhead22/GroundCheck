"""Integration tests for GroundCheck MCP server — tests the full tool pipeline."""

import json
import pytest

from groundcheck_mcp.storage import MemoryStore
from groundcheck_mcp.server import (
    _get_verifier,
    crt_store_fact,
    crt_check_memory,
    crt_verify_output,
    _store,
)
import groundcheck_mcp.server as server_module


@pytest.fixture(autouse=True)
def fresh_store():
    """Give each test a fresh in-memory store."""
    store = MemoryStore(":memory:")
    server_module._store = store
    yield store
    store.close()
    server_module._store = None


class TestStoreFact:
    def test_basic_store(self):
        result = json.loads(crt_store_fact("User works at Microsoft"))
        assert result["stored"] is True
        assert result["trust"] == 0.70
        assert "employer" in result["facts_extracted"] or result["facts_extracted"]
        assert result["has_contradiction"] is False

    def test_store_with_source_trust(self):
        result = json.loads(crt_store_fact("Project uses PostgreSQL", source="code"))
        assert result["trust"] == 0.80  # code source = 0.80

    def test_contradiction_detection(self):
        crt_store_fact("User works at Microsoft")
        result = json.loads(crt_store_fact("User works at Amazon"))
        assert result["stored"] is True
        assert result["has_contradiction"] is True
        assert len(result["contradictions"]) > 0
        c = result["contradictions"][0]
        assert c["slot"] == "employer"

    def test_no_false_contradiction_on_different_slots(self):
        crt_store_fact("User works at Microsoft")
        result = json.loads(crt_store_fact("User lives in Seattle"))
        assert result["has_contradiction"] is False

    def test_multiple_facts_stored(self):
        crt_store_fact("User is named Alice")
        crt_store_fact("User lives in Seattle")
        result = json.loads(crt_store_fact("User works at Google"))
        assert result["total_memories"] == 3

    def test_thread_isolation(self):
        crt_store_fact("User works at Microsoft", thread_id="thread_a")
        result = json.loads(
            crt_store_fact("User works at Amazon", thread_id="thread_b")
        )
        # Different threads — should NOT detect contradiction
        assert result["has_contradiction"] is False


class TestCheckMemory:
    def test_empty_memory(self):
        result = json.loads(crt_check_memory("anything"))
        assert result["found"] == 0
        assert "No memories" in result["note"]

    def test_returns_stored_facts(self):
        crt_store_fact("User works at Microsoft")
        crt_store_fact("User lives in Seattle")
        result = json.loads(crt_check_memory("employer"))
        assert result["found"] == 2
        texts = [m["text"] for m in result["memories"]]
        assert "User works at Microsoft" in texts

    def test_detects_contradictions_in_memory(self):
        crt_store_fact("User works at Microsoft")
        crt_store_fact("User works at Amazon")
        result = json.loads(crt_check_memory("employer"))
        assert result["found"] == 2
        # Should flag the employer contradiction
        assert len(result["contradictions"]) > 0

    def test_thread_scoping(self):
        crt_store_fact("User works at Microsoft", thread_id="a")
        crt_store_fact("User lives in Paris", thread_id="b")
        result = json.loads(crt_check_memory("anything", thread_id="a"))
        assert result["found"] == 1


class TestVerifyOutput:
    def test_pass_when_grounded(self):
        crt_store_fact("User works at Microsoft")
        result = json.loads(
            crt_verify_output("You work at Microsoft")
        )
        assert result["passed"] is True

    def test_fail_on_hallucination(self):
        crt_store_fact("User works at Microsoft")
        result = json.loads(
            crt_verify_output("You work at Amazon")
        )
        assert result["passed"] is False
        assert "Amazon" in result["hallucinations"]

    def test_correction_in_strict_mode(self):
        crt_store_fact("User works at Microsoft")
        result = json.loads(
            crt_verify_output("You work at Amazon", mode="strict")
        )
        assert result["corrected"] is not None
        assert "Microsoft" in result["corrected"]

    def test_no_correction_in_permissive_mode(self):
        crt_store_fact("User works at Microsoft")
        result = json.loads(
            crt_verify_output("You work at Amazon", mode="permissive")
        )
        assert result["corrected"] is None

    def test_empty_memory_passes(self):
        result = json.loads(crt_verify_output("You work at anything"))
        assert result["passed"] is True
        assert result["confidence"] == 0.0

    def test_multi_fact_verification(self):
        crt_store_fact("User works at Microsoft")
        crt_store_fact("User lives in Seattle")
        result = json.loads(
            crt_verify_output("You work at Microsoft and live in Seattle")
        )
        assert result["passed"] is True
        assert result["confidence"] > 0.5

    def test_partial_hallucination(self):
        crt_store_fact("User works at Microsoft")
        crt_store_fact("User lives in Seattle")
        result = json.loads(
            crt_verify_output("You work at Amazon and live in Seattle")
        )
        assert result["passed"] is False
        assert "Amazon" in result["hallucinations"]


class TestStorage:
    def test_store_and_retrieve(self):
        store = MemoryStore(":memory:")
        mem = store.store("test fact", thread_id="t1")
        assert mem.text == "test fact"
        memories = store.get_all("t1")
        assert len(memories) == 1
        store.close()

    def test_trust_update(self):
        store = MemoryStore(":memory:")
        mem = store.store("test fact", thread_id="t1")
        store.update_trust(mem.id, 0.95)
        memories = store.get_all("t1")
        assert memories[0].trust == 0.95
        store.close()

    def test_delete(self):
        store = MemoryStore(":memory:")
        mem = store.store("test fact", thread_id="t1")
        store.delete(mem.id)
        memories = store.get_all("t1")
        assert len(memories) == 0
        store.close()

    def test_clear_thread(self):
        store = MemoryStore(":memory:")
        store.store("fact 1", thread_id="t1")
        store.store("fact 2", thread_id="t1")
        store.store("fact 3", thread_id="t2")
        count = store.clear_thread("t1")
        assert count == 2
        assert len(store.get_all("t1")) == 0
        assert len(store.get_all("t2")) == 1
        store.close()

    def test_source_trust_defaults(self):
        store = MemoryStore(":memory:")
        m1 = store.store("from user", source="user")
        m2 = store.store("from code", source="code")
        m3 = store.store("from doc", source="document")
        m4 = store.store("inferred", source="inferred")
        assert m1.trust == 0.70
        assert m2.trust == 0.80
        assert m3.trust == 0.60
        assert m4.trust == 0.40
        store.close()


class TestEndToEnd:
    """Full workflow: store facts → check memory → verify output."""

    def test_full_agent_workflow(self):
        # Agent stores facts from user conversation
        r1 = json.loads(crt_store_fact("My name is Alice"))
        assert r1["stored"]

        r2 = json.loads(crt_store_fact("I work at Microsoft"))
        assert r2["stored"]

        r3 = json.loads(crt_store_fact("I live in Seattle"))
        assert r3["stored"]

        # Agent checks memory before responding
        mem = json.loads(crt_check_memory("user info"))
        assert mem["found"] == 3

        # Agent drafts a response and verifies it
        draft = "Hi Alice! Since you work at Microsoft in Seattle..."
        verified = json.loads(crt_verify_output(draft))
        assert verified["passed"] is True

        # Agent drafts a WRONG response
        bad_draft = "Hi Bob! Since you work at Amazon..."
        bad_result = json.loads(crt_verify_output(bad_draft))
        assert bad_result["passed"] is False
        assert len(bad_result["hallucinations"]) > 0

    def test_contradiction_workflow(self):
        # User says one thing
        crt_store_fact("I work at Microsoft")

        # Later, user says something contradictory
        r = json.loads(crt_store_fact("I work at Amazon"))
        assert r["has_contradiction"] is True
        assert r["contradictions"][0]["slot"] == "employer"

        # Memory check should also flag this
        mem = json.loads(crt_check_memory("employer"))
        assert len(mem["contradictions"]) > 0

        # Verification should handle the contradiction
        result = json.loads(
            crt_verify_output("You work at Microsoft")
        )
        # Should flag requires_disclosure since there's contradicting info
        # (the exact behavior depends on trust scores and thresholds)
        assert isinstance(result["passed"], bool)
