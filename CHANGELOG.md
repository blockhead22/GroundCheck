# Changelog

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
