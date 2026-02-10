# Changelog

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
