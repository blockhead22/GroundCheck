# GroundCheck

**Trust-weighted hallucination detection for AI agents. Zero dependencies. Sub-2ms.**

[![PyPI version](https://badge.fury.io/py/groundcheck.svg)](https://pypi.org/project/groundcheck/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)]()

---

## The Problem

Your AI agent says "you work at Amazon." Memory says "Microsoft." Most systems won't catch this — they just return the most similar embedding and hope for the best. GroundCheck catches it in <2ms with zero dependencies.

## Install

```bash
pip install groundcheck
```

## 10-Second Demo

```python
from groundcheck import GroundCheck, Memory

verifier = GroundCheck()

memories = [
    Memory(id="m1", text="User works at Microsoft", trust=0.9),
    Memory(id="m2", text="User lives in Seattle", trust=0.8),
]

result = verifier.verify("You work at Amazon and live in Seattle", memories)

print(result.passed)          # False
print(result.hallucinations)  # ["Amazon"]
print(result.corrected)       # "You work at Microsoft and live in Seattle"
print(result.confidence)      # 0.65
```

## What Makes This Different

Other systems treat verification as a binary "is this grounded?" check against a single source. GroundCheck is different:

| | Other systems | GroundCheck |
|---|---|---|
| Sources | Single string or premise/hypothesis pair | Multiple memories with per-source trust scores |
| Trust | All sources treated equally | Trust-weighted — high-trust memories override low-trust |
| Contradictions | Not detected | Cross-memory conflict detection with resolution |
| Correction | Flag only — no fix | Auto-rewrites hallucinations with grounded facts |
| Temporal | No awareness | `most_recent` vs `most_trusted` resolution |
| Dependencies | Often torch, transformers, etc. | **Zero** (stdlib only) |
| Latency | 500ms – 3,000ms+ | **1.17ms mean** |
| Extra LLM calls | Some require 3-5 per check | **Zero** |

## How It Works

```
Generated text + Retrieved memories (with trust scores)
    → Extract fact claims (slot-based: name, employer, location, ...)
    → Detect contradictions across memories
    → Build grounding map (fuzzy match claims to memories)
    → Check disclosure requirements (trust-weighted)
    → Calculate confidence score
    → Generate corrections (strict mode)
    → VerificationReport
```

## Trust-Weighted Verification

GroundCheck doesn't treat all sources equally. Each memory has a trust score:

```python
memories = [
    Memory(id="m1", text="User is named Alice", trust=0.9),   # High trust
    Memory(id="m2", text="User is named Bob", trust=0.3),     # Low trust
]

result = verifier.verify("Your name is Bob", memories)
print(result.requires_disclosure)  # True — trust gap > 0.3
print(result.contradiction_details[0].most_trusted_value)  # "alice"
print(result.contradiction_details[0].most_recent_value)   # depends on timestamps
```

## Verification Modes

- **`strict`** — generates corrected text, replaces hallucinations with grounded facts
- **`permissive`** — detects and reports, doesn't rewrite

```python
result = verifier.verify("You live in Paris", memories, mode="strict")
print(result.corrected)  # Rewritten with grounded facts

result = verifier.verify("You live in Paris", memories, mode="permissive")
print(result.corrected)  # None — permissive doesn't rewrite
```

## Supported Fact Slots

15+ built-in slot types with mutual exclusivity knowledge:

`name`, `employer`, `location`, `title`, `occupation`, `age`, `school`,
`degree`, `favorite_color`, `coffee`, `hobby`, `pet`, `project`,
`graduation_year`, `programming_experience`, and more.

GroundCheck knows that a person can only have one employer at a time, but can have
multiple hobbies. This built-in domain knowledge prevents false positives.

## Neural Mode (Optional)

For paraphrase handling and semantic matching:

```bash
pip install groundcheck[neural]
```

```python
# Automatically used when sentence-transformers is installed
verifier = GroundCheck()  # Detects neural availability
result = verifier.verify("Employed by Google", memories)  # Matches "works at Google"
```

| Mode | Paraphrase Accuracy | Latency |
|------|-------------------|---------|
| Regex-only (default) | 70% | 1.17ms |
| Neural | 85-90% | ~15ms |

## API Reference

### `GroundCheck`
- `verify(generated_text, retrieved_memories, mode="strict")` → `VerificationReport`
- `extract_claims(text)` → `Dict[str, ExtractedFact]`
- `find_support(claim, memories)` → match info

### `VerificationReport`
- `passed: bool` — did verification pass?
- `corrected: Optional[str]` — rewritten text (strict mode)
- `hallucinations: List[str]` — hallucinated values
- `grounding_map: Dict` — claim → supporting memory
- `confidence: float` — trust-weighted confidence (0.0-1.0)
- `contradiction_details: List[ContradictionDetail]` — full conflict info
- `requires_disclosure: bool` — must the response acknowledge conflicts?

### `Memory`
- `id: str` — unique identifier
- `text: str` — memory content
- `trust: float` — trust score (0.0-1.0, default 1.0)
- `timestamp: Optional[int]` — when this was stored

### `ContradictionDetail`
- `slot: str` — which fact slot conflicts
- `values: List[str]` — conflicting values
- `most_trusted_value` — value from highest-trust memory
- `most_recent_value` — value from most recent memory

## Performance

```
Benchmark: 1,000 verifications
Mean latency:  1.17ms
P95 latency:   2.09ms
P99 latency:   3.41ms
Memory: ~2MB RSS
Dependencies: 0
```

## MCP Server (Agent Integration)

GroundCheck ships with an MCP server that gives any AI agent (Copilot, Claude, Cursor) persistent fact memory with contradiction detection:

```bash
pip install groundcheck[mcp]
groundcheck-mcp --db .groundcheck/memory.db
```

Add to VS Code's MCP config:
```json
{
  "servers": {
    "groundcheck": {
      "command": "groundcheck-mcp",
      "args": ["--db", ".groundcheck/memory.db"]
    }
  }
}
```

Tools exposed: `crt_store_fact`, `crt_check_memory`, `crt_verify_output`.

## Development

```bash
git clone https://github.com/blockhead22/GroundCheck.git
cd GroundCheck
python -m venv .venv
.venv\Scripts\activate  # or source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## License

MIT
