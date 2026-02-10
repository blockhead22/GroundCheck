# GroundCheck

**Trust-weighted hallucination detection for AI agents. Zero dependencies. Sub-2ms.**

[![PyPI version](https://badge.fury.io/py/groundcheck.svg)](https://pypi.org/project/groundcheck/)
[![CI](https://github.com/blockhead22/GroundCheck/actions/workflows/groundcheck-test.yml/badge.svg)](https://github.com/blockhead22/GroundCheck/actions/workflows/groundcheck-test.yml)
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
    → Extract fact claims (universal: any domain, any structure)
    → Detect contradictions across memories (dynamic slot tracking)
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

## Universal Fact Extraction

GroundCheck v0.2 extracts facts from **any domain** — not just personal profiles. Nine pattern families cover:

| Pattern | Example |
|---|---|
| Copular (`X is Y`) | "The server is running Ubuntu 22.04" |
| Possessive (`X has Y`) | "Python has garbage collection" |
| Non-copular verbs | "Tesla manufactures electric vehicles" |
| Clause splitting | "Bob is 30, lives in NYC, and works at Google" |
| Decisions & plans | "We chose Postgres" / "They decided to use Rust" |
| Requirements | "The app requires Node 18+" |
| Prescriptive | "Always use HTTPS for API calls" |
| Numeric | "The latency is 3.5ms" / "Revenue: $4.2 billion" |
| Named slots | `name`, `employer`, `location`, `age`, `hobby`, etc. |

35+ known exclusive slots with mutual exclusivity knowledge (a person has one employer but many hobbies). **All extracted slots** are tracked for contradictions — including dynamically discovered ones.

## Neural Mode (Optional)

For paraphrase handling and semantic matching, install the neural extras:

```bash
pip install groundcheck[neural]
```

```python
# Explicit control (v0.3.0+)
verifier = GroundCheck(neural=True)   # Enable paraphrase matching (default)
verifier = GroundCheck(neural=False)  # Zero-dep, sub-2ms regex only

# Catches paraphrases regex can't:
memories = [Memory(id="m1", text="User works at Google")]
result = verifier.verify("Employed by Google", memories)   # ✓ passes
result = verifier.verify("I live in New York City",        # ✓ matches "NYC"
         [Memory(id="m2", text="User lives in NYC")])
```

Models are loaded **lazily** on first use — no startup cost until you need them.

Five matching strategies fire in order: exact → normalization → fuzzy → synonym → embedding.
NLI-based contradiction refinement filters false positives for dynamically-discovered slots.

| Mode | Paraphrase Accuracy | Latency |
|------|-------------------|---------|
| Regex-only (default) | 70% | 1.17ms |
| Neural | 85-90% | ~15ms |

## API Reference

### `GroundCheck`
- `GroundCheck(neural=True)` — constructor. `neural=True` enables semantic matching (requires `groundcheck[neural]`), `neural=False` for zero-dependency mode.
- `verify(generated_text, retrieved_memories, mode="strict")` → `VerificationReport`
- `extract_claims(text)` → `Dict[str, ExtractedFact]`
- `find_support(claim, memories)` → match info

### `extract_fact_slots(text)` (standalone function)
Universal fact extractor — works on any domain text, not just personal facts.
Returns `Dict[str, ExtractedFact]` with dynamically discovered slot names.

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

GroundCheck ships with an MCP server that gives any AI agent persistent fact memory with contradiction detection. Works with **VS Code Copilot**, **Claude Desktop**, **Cursor**, and any MCP-compatible client.

```bash
pip install groundcheck[mcp]
```

Add to your config (VS Code `.vscode/mcp.json`, Claude `claude_desktop_config.json`, etc.):

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

Three tools are exposed:

| Tool | When to call | What it does |
|------|-------------|-------------|
| `crt_store_fact` | User states a fact | Stores with trust score, detects contradictions |
| `crt_check_memory` | Before answering about the user | Returns relevant memories with trust scores |
| `crt_verify_output` | Before sending a response | Catches hallucinations, auto-corrects, scores confidence |

**[Full MCP setup guide →](docs/mcp-server.md)**

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
