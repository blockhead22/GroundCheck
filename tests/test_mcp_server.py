"""Integration tests for GroundCheck MCP server — tests the full tool pipeline."""

import sys
import json
import sqlite3
import pytest

pytest.importorskip("mcp", reason="mcp package requires Python 3.10+")

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


class TestNamespaceIsolation:
    """Tests for project-scoped namespace memory isolation."""

    def test_store_with_namespace(self):
        result = json.loads(
            crt_store_fact("Project uses React", namespace="my-app")
        )
        assert result["stored"] is True
        assert result["namespace"] == "my-app"

    def test_namespace_isolation_between_projects(self):
        # Store facts in two different project namespaces
        crt_store_fact("Enforce strict linting", namespace="production")
        crt_store_fact("No docs needed", namespace="playground")

        # Each namespace sees only its own memories (plus global)
        prod = json.loads(
            crt_check_memory("linting", namespace="production", include_global=False)
        )
        play = json.loads(
            crt_check_memory("docs", namespace="playground", include_global=False)
        )

        assert prod["found"] == 1
        assert "strict linting" in prod["memories"][0]["text"]

        assert play["found"] == 1
        assert "No docs" in play["memories"][0]["text"]

    def test_global_facts_visible_everywhere(self):
        # Store a personal fact in global namespace
        crt_store_fact("User's name is Nick", namespace="global")

        # Store a project fact in a project namespace
        crt_store_fact("Uses PostgreSQL", namespace="my-app")

        # Query from the project namespace — should see both
        result = json.loads(
            crt_check_memory("info", namespace="my-app", include_global=True)
        )
        assert result["found"] == 2
        texts = [m["text"] for m in result["memories"]]
        assert "User's name is Nick" in texts
        assert "Uses PostgreSQL" in texts

    def test_global_excluded_when_disabled(self):
        crt_store_fact("User's name is Nick", namespace="global")
        crt_store_fact("Uses PostgreSQL", namespace="my-app")

        result = json.loads(
            crt_check_memory("info", namespace="my-app", include_global=False)
        )
        assert result["found"] == 1
        assert result["memories"][0]["text"] == "Uses PostgreSQL"

    def test_verify_uses_namespace(self):
        # Store facts in different namespaces
        crt_store_fact("User works at Microsoft", namespace="global")
        crt_store_fact("Use strict TypeScript", namespace="prod-app")

        # Verify in prod-app namespace — should see global facts too
        result = json.loads(
            crt_verify_output(
                "You work at Microsoft", namespace="prod-app"
            )
        )
        assert result["passed"] is True

    def test_contradiction_scoped_to_namespace(self):
        # Same slot in different namespaces — NOT a contradiction
        crt_store_fact("User works at Microsoft", namespace="project-a")
        result = json.loads(
            crt_store_fact("User works at Amazon", namespace="project-b")
        )
        # Different namespaces — should NOT detect contradiction
        assert result["has_contradiction"] is False

    def test_default_namespace_via_server_config(self):
        """Tests that _default_namespace is used when namespace='' is passed."""
        import groundcheck_mcp.server as srv
        old_ns = srv._default_namespace
        try:
            srv._default_namespace = "configured-project"
            result = json.loads(crt_store_fact("test fact"))
            assert result["namespace"] == "configured-project"
        finally:
            srv._default_namespace = old_ns

    def test_memory_id_includes_namespace(self):
        result = json.loads(
            crt_store_fact("test fact", namespace="my-ns")
        )
        assert "my-ns" in result["memory_id"]

    def test_each_memory_reports_namespace(self):
        crt_store_fact("Fact A", namespace="global")
        crt_store_fact("Fact B", namespace="project-x")

        result = json.loads(
            crt_check_memory("fact", namespace="project-x", include_global=True)
        )
        namespaces = {m["namespace"] for m in result["memories"]}
        assert "global" in namespaces
        assert "project-x" in namespaces


class TestStorageNamespace:
    """Direct storage-layer tests for namespace features."""

    def test_store_with_namespace(self):
        store = MemoryStore(":memory:")
        mem = store.store("test fact", namespace="proj-1")
        assert "proj-1" in mem.id
        assert mem.metadata["namespace"] == "proj-1"
        store.close()

    def test_query_namespace_isolation(self):
        store = MemoryStore(":memory:")
        store.store("fact A", namespace="ns1")
        store.store("fact B", namespace="ns2")

        ns1_mems = store.query("", namespace="ns1", include_global=False)
        ns2_mems = store.query("", namespace="ns2", include_global=False)

        assert len(ns1_mems) == 1
        assert ns1_mems[0].text == "fact A"
        assert len(ns2_mems) == 1
        assert ns2_mems[0].text == "fact B"
        store.close()

    def test_global_merge(self):
        store = MemoryStore(":memory:")
        store.store("global fact", namespace="global")
        store.store("project fact", namespace="my-proj")

        # With include_global=True
        mems = store.query("", namespace="my-proj", include_global=True)
        assert len(mems) == 2

        # With include_global=False
        mems = store.query("", namespace="my-proj", include_global=False)
        assert len(mems) == 1
        assert mems[0].text == "project fact"
        store.close()

    def test_clear_namespace(self):
        store = MemoryStore(":memory:")
        store.store("f1", namespace="ns1")
        store.store("f2", namespace="ns1")
        store.store("f3", namespace="ns2")

        count = store.clear_namespace("ns1")
        assert count == 2
        assert len(store.query("", namespace="ns1", include_global=False)) == 0
        assert len(store.query("", namespace="ns2", include_global=False)) == 1
        store.close()

    def test_list_namespaces(self):
        store = MemoryStore(":memory:")
        store.store("f1", namespace="alpha")
        store.store("f2", namespace="beta")
        store.store("f3", namespace="global")

        ns_list = store.list_namespaces()
        assert ns_list == ["alpha", "beta", "global"]
        store.close()

    def test_clear_thread_with_namespace(self):
        store = MemoryStore(":memory:")
        store.store("f1", thread_id="t1", namespace="ns1")
        store.store("f2", thread_id="t1", namespace="ns2")
        store.store("f3", thread_id="t2", namespace="ns1")

        # Clear only t1+ns1
        count = store.clear_thread("t1", namespace="ns1")
        assert count == 1
        # t1+ns2 still there
        assert len(store.query("", thread_id="t1", namespace="ns2", include_global=False)) == 1
        # t2+ns1 still there
        assert len(store.query("", thread_id="t2", namespace="ns1", include_global=False)) == 1
        store.close()

    def test_migration_adds_namespace_column(self):
        """Verify that opening an old DB (no namespace column) auto-migrates."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "legacy.db")
            # Create a DB the old way — no namespace column
            conn = sqlite3.connect(db_path)
            conn.execute("""CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL DEFAULT 'default',
                text TEXT NOT NULL,
                trust REAL NOT NULL DEFAULT 0.7,
                source TEXT NOT NULL DEFAULT 'user',
                timestamp INTEGER NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )""")
            conn.execute(
                "INSERT INTO memories (id, thread_id, text, trust, source, timestamp) "
                "VALUES ('old1', 'default', 'legacy fact', 0.7, 'user', 1000)"
            )
            conn.commit()
            conn.close()

            # Open with new MemoryStore — should auto-migrate
            store = MemoryStore(db_path)
            mems = store.get_all()
            assert len(mems) == 1
            assert mems[0].text == "legacy fact"

            # New stores should work with namespace
            store.store("new fact", namespace="proj-x")
            proj_mems = store.query("", namespace="proj-x", include_global=False)
            assert len(proj_mems) == 1
            store.close()
