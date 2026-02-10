# Release Notes

Use these to create GitHub Releases at:
https://github.com/blockhead22/GroundCheck/releases/new

---

## groundcheck-v0.3.0 — Neural Mode

**Tag:** `groundcheck-v0.3.0`
**Title:** v0.3.0 — Neural Mode

### What's New

**Neural paraphrase detection** — GroundCheck can now match "employed by Google" against "works at Google", "NYC" against "New York City", and "software developer" against "software engineer". Install the optional neural extras to enable:

```bash
pip install groundcheck[neural]
```

**Explicit neural control** — `GroundCheck(neural=True)` enables semantic matching. Pass `neural=False` for zero-dependency sub-2ms mode. Default is `True`.

**Lazy model loading** — Embedding and NLI models load on first use, not at import. Zero startup cost until you need paraphrase matching.

**NLI contradiction refinement** — Dynamically-discovered slot contradictions are now confirmed via Natural Language Inference, reducing false positives.

### Breaking Changes
None. All v0.2.x code is backward compatible.

### Stats
- 302 tests passing (18 new for neural integration)
- All 16 previous neural tests now run (0 skipped)
- Still zero runtime dependencies for core mode

### Install
```bash
pip install groundcheck==0.3.0          # Core only
pip install "groundcheck[neural]==0.3.0" # With paraphrase matching
```

---

## groundcheck-v0.2.0 — Universal Extraction

**Tag:** `groundcheck-v0.2.0`
**Title:** v0.2.0 — Universal Fact Extraction

### What's New

**Universal Extraction Engine** — GroundCheck is no longer limited to personal-profile facts. Nine pattern families now extract facts from any domain:

- **Copular patterns** (`X is Y`): "The server is running Ubuntu 22.04"
- **Possessive patterns** (`X has Y`): "Python has garbage collection"
- **Non-copular verbs**: "Tesla manufactures electric vehicles"
- **Clause splitting**: "Bob is 30, lives in NYC, and works at Google" → 3 separate facts
- **Decisions & plans**: "We chose Postgres" / "They decided to use Rust"
- **Requirements**: "The app requires Node 18+"
- **Prescriptive rules**: "Always use HTTPS for API calls"
- **Numeric-safe capture**: Handles decimals, currencies, percentages without truncation
- **Named slots**: All original personal-fact slots preserved

**Dynamic Contradiction Detection** — All extracted slots (including dynamically discovered ones) are now tracked for contradictions. 35+ known exclusive slots with mutual exclusivity knowledge.

### Stats
- 280 tests passing (26 new for universal extraction)
- Zero breaking changes — all v0.1.x code works unchanged
- Still zero dependencies, still sub-2ms

### Install
```bash
pip install groundcheck==0.2.0
```

---

## groundcheck-v0.1.1 — Initial PyPI Release

**Tag:** `groundcheck-v0.1.1`
**Title:** v0.1.1 — Initial Release

### What's New

First public release of GroundCheck on PyPI.

- **Trust-weighted verification** — each memory has a trust score, high-trust overrides low-trust
- **Contradiction detection** — cross-memory conflict detection with most-trusted/most-recent resolution
- **Auto-correction** — strict mode rewrites hallucinations with grounded facts
- **15+ fact slots** — name, employer, location, age, hobby, and more with mutual exclusivity knowledge
- **MCP server** — `pip install groundcheck[mcp]` for AI agent integration
- **Neural mode** — optional `pip install groundcheck[neural]` for paraphrase handling
- **Zero dependencies** — stdlib only for core, sub-2ms latency

### Install
```bash
pip install groundcheck==0.1.1
```
