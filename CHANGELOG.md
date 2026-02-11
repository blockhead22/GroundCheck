# Changelog

## [0.3.0] - 2026-02-10

### Added
- **`neural=True` parameter** — `GroundCheck(neural=True)` explicitly controls whether semantic matching is active. Defaults to `True` for backward compatibility; pass `neural=False` for zero-dependency sub-2ms mode.
- **Lazy model loading** — Embedding models (all-MiniLM-L6-v2) and NLI models (nli-deberta-v3-small) are loaded lazily on first use, not at import time. Zero startup cost until neural features are needed.
- **SemanticContradictionDetector integration** — NLI-based contradiction refinement for dynamically-discovered slots. Known-exclusive slots still use fast slot-based detection; dynamic slots get NLI confirmation to reduce false positives.
- **18 new neural integration tests** — End-to-end tests proving:
  - Paraphrase detection: "employed by Google" ↔ "works at Google"
  - Location normalization: "resides in Seattle" ↔ "lives in Seattle"  
  - Education paraphrases: "studied at MIT" ↔ "graduated from MIT"
  - Value-level matching: "NYC" ↔ "New York City" (embedding similarity)
  - Synonym expansion: "software developer" ↔ "software engineer"
  - Negative tests: unrelated entities still rejected

### Changed
- Removed eager `SentenceTransformer` loading from `GroundCheck.__init__()` — the `SemanticMatcher` handles lazy loading, eliminating redundant ~500ms startup overhead
- Removed dead inline embedding fallback code from `_is_value_supported()` — all embedding logic now lives in `SemanticMatcher`
- `_detect_contradictions()` now uses NLI to confirm dynamic-slot contradictions before flagging them

### Fixed
- Neural mode no longer loads models at import time, reducing memory usage for users who don't need embeddings
- False positive contradictions in dynamic (non-hardcoded) slots are now filtered by NLI confidence

## [0.2.0] - 2026-02-10

### Added
- **Universal fact extraction** — 9 pattern families in catch-all extractor:
  - Non-copular verbs: `X uses/handles/supports/runs Y`
  - Requirements: `X requires/needs/demands Y`
  - Decisions: `We agreed/decided to use X`
  - Prescriptive: `X should be/must be/needs to be Y`
  - Passive voice: `X is handled/managed/done via Y`
  - Bare-subject copular: `Backend is FastAPI` (no article needed)
  - Config: `X is set to/configured as Y`
  - Equality: `X equals Y`
- **Clause splitting** — commas/semicolons split into sub-clauses for multi-fact extraction
- **Dynamic contradiction detection** — ANY extracted slot can trigger contradictions, not just the 35 hardcoded ones
- `ADDITIVE_SLOTS` set for slots where multiple values are not contradictions (skill, language, tool, etc.)
- `KNOWN_EXCLUSIVE_SLOTS` replaces `MUTUALLY_EXCLUSIVE_SLOTS` (backward-compatible alias kept)
- Decimal-safe value capture (periods in `99.9%`, `v3.11` no longer truncate)
- Possessive pronoun stripping (`your goal` → `goal`, `my project` → `project`)
- Question word blocklist prevents false extraction from interrogative sentences
- 26 new tests covering all 10 audit sentences + dynamic contradictions

### Fixed
- 6 of 10 real-world declarative sentences were silently dropped by the old extractor
- Comma-separated compound sentences like "frontend is React, backend is FastAPI" now extract both facts
- Values containing decimal numbers (99.9%, v3.11) no longer truncated at the period
- Single-character values like "5" in "Max retries should be 5" now extracted

## [0.1.1] - 2026-02-09

### Fixed
- Corrected GitHub repository URLs in pyproject.toml
- Removed competitor references from README

## [0.1.0] - 2026-02-09

### Added
- Core `GroundCheck` verifier with strict/permissive modes
- `Memory` dataclass with trust scoring and temporal weighting
- `VerificationReport` with hallucination detection + auto-correction
- `ContradictionDetail` with `most_recent_value` / `most_trusted_value`
- `extract_fact_slots()` — 15+ slot types from natural language
- Zero runtime dependencies (stdlib only)
- Optional neural verification via `pip install groundcheck[neural]`
- MCP server for AI agent integration via `pip install groundcheck[mcp]`
- 173 tests, sub-second execution
- GitHub Actions CI (Python 3.9–3.13)
- PyPI trusted publisher workflow
