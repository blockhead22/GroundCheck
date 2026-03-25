# GroundCheck v2.0.0 Extraction Session — CRT Trust & Contradiction Engine

## Role
You are extracting the CRT (Contradiction-aware Reasoning & Trust) system from a monolith at `D:/AI_round2` into the standalone GroundCheck library at `D:/groundcheck`. The library already has v1.0.0 published (verification, fact extraction, MCP server). You are adding the trust math, contradiction ledger, lifecycle engine, and ML detection layer — the pieces that make this novel.

## CRITICAL RULES
1. **PAUSE after each numbered step.** Print `--- STEP N COMPLETE. Ready for next step? ---` and WAIT for the user to say "go" or "next" before continuing.
2. **Do NOT modify D:/AI_round2.** Read-only source. Copy and adapt, never edit the original.
3. **Preserve the zero-dependency core principle.** New modules that need numpy/sklearn go behind optional imports with graceful fallback.
4. **All new code goes under `D:/groundcheck/groundcheck/`** in the existing package structure.
5. **Run tests after every step that adds code.** Never move to the next step with failing tests.
6. **Keep existing v1.0 API intact.** Nothing that works today should break.

---

## SOURCE FILES (read-only, in D:/AI_round2/personal_agent/)

| File | LOC | What to extract | Dependencies |
|------|-----|-----------------|-------------|
| `crt_core.py` | 1228 | CRTConfig, CRTMath, trust evolution equations, drift/similarity, contradiction detection rules | numpy (optional) |
| `contradiction_lifecycle.py` | 596 | ContradictionLifecycleState, ContradictionLifecycle, DisclosurePolicy, TransparencyLevel, UserTransparencyPrefs | None (pure stdlib) |
| `contradiction_trace_logger.py` | 331 | ContradictionTraceLogger | None (pure stdlib) |
| `crt_ledger.py` | 1417 | ContradictionLedger, ContradictionEntry, ContradictionStatus, ContradictionType | sqlite3, crt_core |
| `ml_contradiction_detector.py` | 678 | MLContradictionDetector, retraction/equivalence/enrichment detection | sklearn (optional) |
| `trust_decay.py` | 388 | Trust decay pass, exponential decay, drift-aware reinforcement | crt_core, sqlite3 |

## EXISTING FILES (in D:/groundcheck/groundcheck/ — do NOT break these)

| File | Purpose |
|------|---------|
| `__init__.py` | Public API exports |
| `types.py` | Memory, VerificationReport, ExtractedFact, ContradictionDetail |
| `verifier.py` | GroundCheck class — core verification |
| `fact_extractor.py` | Regex fact extraction |
| `knowledge_extractor.py` | Tier 1.5 verb ontology extraction |
| `semantic_contradiction.py` | NLI-based contradiction detection |
| `semantic_matcher.py` | Embedding similarity matching |
| `neural_extractor.py` | Hybrid neural+regex extraction |
| `cli.py` | CLI entry point |
| `utils.py` | Shared utilities |
| `crt_rag.py` | (legacy, may need updating) |

---

## STEP-BY-STEP PLAN

### STEP 1: Extract CRT Core Math (crt_core.py → trust_math.py)

**Source:** `D:/AI_round2/personal_agent/crt_core.py`
**Target:** `D:/groundcheck/groundcheck/trust_math.py`

Extract and adapt:
- `CRTConfig` dataclass (all thresholds, configurable)
- `CRTMath` class:
  - `similarity(a, b)` — cosine similarity
  - `drift_meaning(old_vec, new_vec)` — semantic drift = 1 - similarity
  - `belief_weight(trust, confidence, alpha=0.7)` — weighted trust+confidence
  - `recency_weight(timestamp, lambda_time)` — exponential recency
  - `retrieval_score(similarity, recency, belief_weight)` — combined score
  - `evolve_trust_aligned(trust, drift)` — trust increase on alignment
  - `evolve_trust_reinforced(trust, drift)` — trust boost on validation
  - `evolve_trust_contradicted(trust, drift)` — trust penalty on contradiction
  - `detect_contradiction(old_text, new_text, old_vec, new_vec, ...)` — the 6-rule detection
  - `compute_volatility(contradiction_entries)` — volatility metric
  - `should_reflect(volatility, theta)` — reflection trigger
- `SSEMode` enum
- `MemorySource` enum

**Adaptation required:**
- Remove any imports from `personal_agent.*`
- Make numpy optional: `try: import numpy ... except ImportError: # fallback to list math`
- `load_from_calibration()` should accept a path argument, not hardcode `artifacts/`
- Keep all constants and thresholds as-is (they're calibrated)

**Test file:** `D:/groundcheck/tests/test_trust_math.py`
```
Test cases:
- trust evolution: aligned increases trust, contradicted decreases
- trust bounds: never below 0.20, never above 0.95
- drift calculation: identical = 0.0, orthogonal = 1.0
- belief_weight: alpha=0.7 gives 70% trust, 30% confidence
- detect_contradiction: all 6 rules (entity swap, negation, boolean inversion, paraphrase tolerance, high drift, confidence drop)
- config defaults match expected values
- config from dict override
```

**PAUSE after this step.**

---

### STEP 2: Extract Contradiction Lifecycle (contradiction_lifecycle.py → lifecycle.py)

**Source:** `D:/AI_round2/personal_agent/contradiction_lifecycle.py`
**Target:** `D:/groundcheck/groundcheck/lifecycle.py`

Extract and adapt:
- `ContradictionLifecycleState` enum (ACTIVE, SETTLING, SETTLED, ARCHIVED)
- `ContradictionLifecycleEntry` dataclass
- `ContradictionLifecycle` class (state machine with transition rules)
- `TransparencyLevel` enum (MINIMAL, BALANCED, AUDIT_HEAVY)
- `MemoryStyle` enum (STICKY, NORMAL, FORGETFUL)
- `UserTransparencyPrefs` dataclass
- `DisclosurePolicy` class (should_disclose, get_disclosure_priority, high-stakes detection)

**Adaptation required:**
- Minimal — this file is already standalone
- Update any absolute imports to relative

**Test file:** `D:/groundcheck/tests/test_lifecycle.py`
```
Test cases:
- state transitions: ACTIVE→SETTLING (2 confirmations), SETTLING→SETTLED (5 confirmations), SETTLED→ARCHIVED (30 days)
- disclosure policy: high-stakes attributes always disclosed
- transparency levels: MINIMAL suppresses, AUDIT_HEAVY always shows
- session disclosure limits respected
- stale detection (age-based)
- serialization: to_dict/from_dict roundtrip
```

**PAUSE after this step.**

---

### STEP 3: Extract Contradiction Trace Logger (contradiction_trace_logger.py → trace_logger.py)

**Source:** `D:/AI_round2/personal_agent/contradiction_trace_logger.py`
**Target:** `D:/groundcheck/groundcheck/trace_logger.py`

Extract and adapt:
- `ContradictionTraceLogger` class
- `get_trace_logger()` module function
- `configure_trace_logging()` module function

**Adaptation required:**
- Default log path should be configurable, not hardcoded to `ai_logs/`
- Use `~/.groundcheck/logs/` as default, respect `GROUNDCHECK_LOG_DIR` env var

**Test file:** `D:/groundcheck/tests/test_trace_logger.py`
```
Test cases:
- logger creation and singleton behavior
- log file creation in configured directory
- structured event logging (contradiction detected, resolution attempt, etc.)
- configure_trace_logging() changes behavior
```

**PAUSE after this step.**

---

### STEP 4: Extract Contradiction Ledger (crt_ledger.py → ledger.py)

**Source:** `D:/AI_round2/personal_agent/crt_ledger.py`
**Target:** `D:/groundcheck/groundcheck/ledger.py`

This is the most complex extraction. Extract:
- `ContradictionStatus` (OPEN, REFLECTING, RESOLVED, ACCEPTED)
- `ContradictionType` (REFINEMENT, REVISION, TEMPORAL, CONFLICT, DENIAL)
- `ContradictionEntry` dataclass
- `ContradictionLedger` class:
  - `record_contradiction()`
  - `resolve_contradiction()`
  - `get_open_contradictions()`
  - `get_all_contradictions()`
  - `has_open_contradiction()`
  - `get_contradiction_stats()`
  - `mark_contradiction_asked()`
  - `record_contradiction_user_answer()`
  - `process_lifecycle_transitions()`
  - `queue_reflection()`

**Adaptation required (IMPORTANT):**
- Remove imports: `fact_slots`, `two_tier_facts`, `crt_semantic_anchor`, `llm_drift_assessor`
- Replace `extract_fact_slots()` calls with GroundCheck's existing `fact_extractor.extract_fact_slots()`
- Replace `TwoTierFactSystem` with GroundCheck's existing `knowledge_extractor`
- Remove or stub `SemanticAnchor` / `generate_clarification_prompt` (make optional)
- Remove `llm_drift_assessor` dependency (make optional callback)
- DB path: accept path in constructor, default to `~/.groundcheck/ledger.db`
- The ledger depends on `CRTMath` from step 1 — use the extracted `trust_math` module

**Test file:** `D:/groundcheck/tests/test_ledger.py`
```
Test cases:
- record and retrieve contradiction
- resolve contradiction updates status
- open contradictions filtered correctly
- lifecycle transitions (OPEN→REFLECTING→RESOLVED→ACCEPTED)
- contradiction stats computation
- reflection queue operations
- database creation and schema
- thread_id scoping
- serialization roundtrip (ContradictionEntry to_dict/from_dict)
```

**PAUSE after this step.**

---

### STEP 5: Extract ML Contradiction Detector (ml_contradiction_detector.py → ml_detector.py)

**Source:** `D:/AI_round2/personal_agent/ml_contradiction_detector.py`
**Target:** `D:/groundcheck/groundcheck/ml_detector.py`

Extract:
- `MLContradictionDetector` class
- All helper functions (retraction patterns, semantic equivalents, detail enrichment, transient state)
- Constants: SEMANTIC_EQUIVALENTS, DETAIL_ENRICHMENT_WORDS, TRANSIENT_STATE_WORDS, RETRACTION_PATTERNS

**Adaptation required:**
- Make sklearn optional: `try: from sklearn... except ImportError: _SKLEARN_AVAILABLE = False`
- When sklearn unavailable, fall back to heuristic detection (already has `_fallback_detection`)
- Model file path: accept in constructor, default to `~/.groundcheck/models/`
- Stub `names_are_related` with simple string comparison (was from `fact_slots`)
- Ship WITHOUT the XGBoost model files in the repo (too large). Instead:
  - Include a `download_models()` utility function
  - OR document how to train them
  - Heuristic fallback works without models

**Test file:** `D:/groundcheck/tests/test_ml_detector.py`
```
Test cases:
- retraction pattern detection ("actually no", "wait I meant")
- semantic equivalence ("PhD" = "doctorate", "NY" = "New York")
- detail enrichment ("dog" → "rescue dog" is NOT contradiction)
- transient state detection (mood/energy words)
- fallback detection without sklearn
- full check_contradiction flow (with mocked models if needed)
```

**PAUSE after this step.**

---

### STEP 6: Extract Trust Decay (trust_decay.py → decay.py)

**Source:** `D:/AI_round2/personal_agent/trust_decay.py`
**Target:** `D:/groundcheck/groundcheck/decay.py`

Extract:
- `run_trust_decay_pass()` — adapted to work with GroundCheck's DB schema
- `reinforce_memory()` — individual memory reinforcement
- Exponential decay calculation
- Drift-aware boost calculation
- Constants: DECAY_RATE, TRUST_FLOOR, TRUST_CEILING, GRACE_PERIOD_DAYS

**Adaptation required:**
- Remove `personal_agent.memory_compression` dependency entirely
- Use `trust_math.CRTMath` from step 1 (not personal_agent.crt_core)
- DB path: accept as argument, also check `GROUNDCHECK_DB` env var
- Schema detection: support GroundCheck's existing schema (from MCP server)
- Make the encode_text function pluggable (accept an encoder callback)

**Test file:** `D:/groundcheck/tests/test_decay.py`
```
Test cases:
- exponential decay reduces trust over time
- grace period protects new memories (7 days)
- trust floor (0.20) is respected
- trust ceiling (0.95) is respected
- drift-aware boost increases trust on access
- correction boost is larger than regular boost
- decay pass on in-memory DB
```

**PAUSE after this step.**

---

### STEP 7: Wire Everything Together — Update __init__.py and Public API

**Target:** `D:/groundcheck/groundcheck/__init__.py`

Add new exports:
```python
# Trust math (core CRT engine)
from .trust_math import CRTConfig, CRTMath, SSEMode, MemorySource

# Contradiction ledger
from .ledger import (
    ContradictionLedger, ContradictionEntry,
    ContradictionStatus, ContradictionType,
)

# Lifecycle engine
from .lifecycle import (
    ContradictionLifecycle, ContradictionLifecycleState,
    ContradictionLifecycleEntry, DisclosurePolicy,
    TransparencyLevel, MemoryStyle, UserTransparencyPrefs,
)

# ML detection (optional)
try:
    from .ml_detector import MLContradictionDetector
    _ML_DETECTOR_AVAILABLE = True
except ImportError:
    _ML_DETECTOR_AVAILABLE = False
    MLContradictionDetector = None

# Trust decay
from .decay import run_trust_decay_pass, reinforce_memory

# Trace logging
from .trace_logger import ContradictionTraceLogger, get_trace_logger
```

Update `__all__` to include all new exports.

Update `pyproject.toml`:
- Version: `2.0.0`
- Add optional dependency group: `[project.optional-dependencies] ml = ["scikit-learn>=1.0.0", "xgboost>=1.5.0"]`
- Keep existing `neural` and `mcp` groups

**Test:** Run full test suite `pytest tests/ -v` — ALL tests must pass (old and new).

**PAUSE after this step.**

---

### STEP 8: Write Integration Example — `examples/trust_aware_agent.py`

Create a complete, runnable example showing the full CRT pipeline:

```
Scenario: A chatbot that remembers user facts, detects contradictions,
tracks trust, and decides when to disclose conflicts.

Flow:
1. Store initial memories with trust scores
2. User says something contradictory
3. GroundCheck detects the contradiction
4. Ledger records it
5. Trust evolves (decreases for contradicted memory)
6. Lifecycle tracks the contradiction state
7. Disclosure policy decides whether to tell the user
8. Verifier catches hallucination attempt using contradicted fact
```

This example should be copy-pasteable and run with `python examples/trust_aware_agent.py` using only `pip install groundcheck`.

Also create `examples/openclaw_plugin.py` — a skeleton showing how GroundCheck would plug into Open WebUI as a pipe/filter function.

**PAUSE after this step.**

---

### STEP 9: Update README.md

Rewrite the README to reflect v2.0.0. Structure:

1. **Header** — same badges, update version
2. **The Problem** — expand: "Your AI agent says you work at Amazon. Memory says Microsoft. Other systems silently overwrite. GroundCheck catches it, tracks it, and decides whether to tell you."
3. **Install** — `pip install groundcheck` (core) / `pip install groundcheck[ml]` (XGBoost) / `pip install groundcheck[neural]` (embeddings)
4. **10-Second Demo** — keep existing
5. **Trust-Aware Demo** — NEW section showing trust evolution + contradiction ledger
6. **Architecture** — diagram showing the full pipeline:
   ```
   Input → Fact Extraction (3 tiers) → Contradiction Detection (5 layers)
     → Trust Math (evolution equations) → Contradiction Ledger (append-only)
     → Lifecycle Engine (state machine) → Disclosure Policy → Output
   ```
7. **What Makes This Different** — updated comparison table including trust math, ledger, lifecycle
8. **API Reference** — all new classes documented
9. **MCP Server** — keep existing
10. **CLI** — keep existing
11. **OpenClaw Integration** — NEW section, link to example
12. **Development** — keep existing
13. **License** — MIT

**PAUSE after this step.**

---

### STEP 10: Update CHANGELOG.md and Git

Add to top of CHANGELOG.md:
```markdown
## [2.0.0] - 2026-03-24

### Added
- **Trust Math Engine** (`trust_math.py`) — CRTConfig, CRTMath with trust evolution
  equations (aligned/reinforced/contradicted), semantic drift measurement,
  6-rule contradiction detection, retrieval scoring, volatility computation.
- **Contradiction Ledger** (`ledger.py`) — Append-only SQLite ledger tracking every
  contradiction. No silent overwrites. Stores drift measurements, resolution status,
  and full audit trail.
- **Lifecycle Engine** (`lifecycle.py`) — State machine for contradiction lifecycle:
  ACTIVE → SETTLING → SETTLED → ARCHIVED. Confirmation counting, disclosure policy
  with transparency levels (MINIMAL/BALANCED/AUDIT_HEAVY), high-stakes detection.
- **ML Contradiction Detector** (`ml_detector.py`) — XGBoost-based detection with
  18 features. Includes retraction pattern matching, semantic equivalence database,
  detail enrichment detection, transient state filtering. Falls back to heuristics
  when sklearn unavailable.
- **Trust Decay** (`decay.py`) — Exponential trust decay with 7-day grace period,
  drift-aware reinforcement on memory access, configurable floor/ceiling.
- **Trace Logger** (`trace_logger.py`) — Structured event logging for contradiction
  resolution debugging.
- **Integration example** (`examples/trust_aware_agent.py`) — Full CRT pipeline demo.
- **OpenClaw plugin skeleton** (`examples/openclaw_plugin.py`).
- Optional dependency groups: `groundcheck[ml]` for XGBoost/sklearn.

### Changed
- Version bump to 2.0.0 (new modules are additive, but this is a major capability expansion).
- README rewritten to cover trust math, ledger, and lifecycle features.
```

Git operations:
```bash
cd D:/groundcheck
git add -A
git status  # review what's staged
git commit -m "v2.0.0: Extract CRT trust math, contradiction ledger, lifecycle engine, ML detector, and trust decay from Aether monolith"
git tag v2.0.0
```

Do NOT push. Show the user the commit and let them review before pushing.

**PAUSE after this step.**

---

### STEP 11: Final Verification

Run the complete check:
1. `pytest tests/ -v --tb=short` — all tests pass
2. `pip install -e .` — editable install works
3. `python -c "from groundcheck import GroundCheck, CRTMath, ContradictionLedger, ContradictionLifecycle; print('All imports OK')"` — public API imports clean
4. `python examples/trust_aware_agent.py` — example runs end-to-end
5. `groundcheck verify "You work at Amazon" --memories test_memories.json` — CLI still works
6. `python -m build` — wheel builds without errors

Report results to user. If anything fails, fix it before declaring done.

**SESSION COMPLETE.**

---

## REFERENCE: Dependency Installation Order

```
Step 1: trust_math.py        (no deps)
Step 2: lifecycle.py          (no deps)
Step 3: trace_logger.py       (no deps)
Step 4: ledger.py             (depends on trust_math from step 1)
Step 5: ml_detector.py        (optional sklearn)
Step 6: decay.py              (depends on trust_math from step 1)
Step 7: __init__.py wiring    (depends on all above)
Step 8: examples              (depends on all above)
Step 9: README                (documentation only)
Step 10: CHANGELOG + git      (packaging only)
Step 11: final verification   (validation only)
```

## REFERENCE: What NOT to Extract (stays in Aether monolith)

- `crt_rag.py` — too intertwined, 9000+ lines, depends on IntentRouter/FactStore/ReasoningEngine
- `crt_memory.py` — monolith-specific memory storage layer
- `crt_semantic_anchor.py` — requires LLM calls for clarification prompts
- `two_tier_facts.py` — already partially covered by existing fact_extractor + knowledge_extractor
- `fact_slots.py` — already extracted as fact_extractor.py in groundcheck
- `llm_drift_assessor.py` — requires LLM, make optional callback instead
