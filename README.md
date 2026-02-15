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

## CLI

```bash
# Verify text against a memory file
groundcheck verify "You work at Amazon" --memories memories.json

# Extract facts from text
groundcheck extract "My name is Alice and I work at Google"

# Show version
groundcheck version
```

Memory files are JSON — either a list of objects or `{"memories": [...]}`:

```json
[
  {"text": "User works at Microsoft", "trust": 0.9},
  {"text": "User lives in Seattle", "trust": 0.8}
]
```

## What Makes This Different

| | Other systems | GroundCheck |
|---|---|---|
| Sources | Single string or premise/hypothesis pair | Multiple memories with per-source trust scores |
| Trust | All sources treated equally | Trust-weighted — high-trust memories override low-trust |
| Contradictions | Not detected | Cross-memory conflict detection with resolution |
| Correction | Flag only — no fix | Auto-rewrites hallucinations with grounded facts |
| Temporal | No awareness | `most_recent` vs `most_trusted` resolution |
| Dependencies | Often torch, transformers, etc. | **Zero** (stdlib only, neural optional) |
| Latency | 500ms – 3,000ms+ | **Sub-2ms** (regex mode) |
| Extra LLM calls | Some require 3-5 per check | **Zero** |

## How It Works

```
Generated text + Retrieved memories (with trust scores)
    → Tier 1: Regex fact extraction (15+ named slots)
    → Tier 1.5: Knowledge-based inference (verb ontology + entity taxonomy)
    → Tier 2: Neural paraphrase matching (optional)
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

## Three-Tier Extraction

### Tier 1 — Regex (always active)
15+ named slots (`name`, `employer`, `location`, `age`, etc.) plus 9 universal pattern families:

| Pattern | Example |
|---|---|
| Copular (`X is Y`) | "The server is running Ubuntu 22.04" |
| Non-copular verbs | "Tesla manufactures electric vehicles" |
| Clause splitting | "Bob is 30, lives in NYC, and works at Google" |
| Decisions & plans | "We chose Postgres" / "They decided to use Rust" |
| Requirements | "The app requires Node 18+" |
| Prescriptive | "Always use HTTPS for API calls" |

### Tier 1.5 — Knowledge Inference (always active)
Understands conversational language that regex misses:

```python
from groundcheck import extract_knowledge_facts

facts = extract_knowledge_facts("Yeah we ended up going with Postgres after the whole MySQL disaster")
# → database: postgres (adoption), mysql (deprecation)
```

Powered by:
- **Verb ontology** — 10 semantic categories (adoption, migration, deprecation, tentative…) with ~200 verb phrases
- **Entity taxonomy** — 22 tech categories with ~500 known entities
- **Inference rules** — clause decomposition → entity recognition → verb semantics → fact extraction

Combined benchmark (42 sentences, 65 slots): **F1 = 83.2%** (+44% over regex alone).

### Tier 2 — Neural (optional)

```bash
pip install groundcheck[neural]
```

```python
verifier = GroundCheck(neural=True)   # Enable paraphrase matching

# Catches paraphrases regex can't:
memories = [Memory(id="m1", text="User works at Google")]
result = verifier.verify("Employed by Google", memories)  # ✓ passes
```

Models are loaded **lazily** on first use — no startup cost until you need them.
Five matching strategies: exact → normalization → fuzzy → synonym → embedding.
NLI-based contradiction refinement filters false positives.

## API Reference

### `GroundCheck`
- `GroundCheck(neural=False)` — constructor. `neural=False` (default) for zero-dependency sub-2ms mode. `neural=True` enables semantic matching (requires `groundcheck[neural]`).
- `verify(generated_text, retrieved_memories, mode="strict")` → `VerificationReport`
- `extract_claims(text)` → `Dict[str, ExtractedFact]`
- `find_support(claim, memories)` → match info

### `extract_fact_slots(text)` (standalone function)
Universal regex fact extractor — works on any domain text.
Returns `Dict[str, ExtractedFact]` with dynamically discovered slot names.

### `extract_knowledge_facts(text)` (standalone function)
Knowledge-based inference extractor using verb ontology and entity taxonomy.
Returns `List[KnowledgeFact]` with inferred relationships.

### `VerificationReport`
- `passed: bool` — did verification pass?
- `corrected: Optional[str]` — rewritten text (strict mode)
- `hallucinations: List[str]` — hallucinated values
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
      "args": ["--db", ".groundcheck/memory.db", "--namespace", "my-project"]
    }
  }
}
```

Five tools are exposed:

| Tool | When to call | What it does |
|------|-------------|-------------|
| `groundcheck_store` | User states a fact | Stores with trust score, detects contradictions |
| `groundcheck_check` | Start of every turn | Returns relevant memories, auto-learns from context |
| `groundcheck_verify` | Before sending a response | Catches hallucinations, auto-corrects, scores confidence |
| `groundcheck_list` | Inspecting memory state | Lists all stored memories with trust scores |
| `groundcheck_delete` | Removing outdated facts | Deletes specific memories or clears thread/namespace |

Memories are scoped by **namespace** so each project gets its own memory. User-level facts stored with `namespace='global'` are visible across all projects.

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
