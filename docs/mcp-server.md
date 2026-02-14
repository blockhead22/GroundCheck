# GroundCheck MCP Server

Give any AI agent persistent, trust-weighted memory with hallucination detection.

The MCP (Model Context Protocol) server exposes GroundCheck's verification engine as three tools that work with **VS Code Copilot**, **Claude Desktop**, **Cursor**, and any MCP-compatible client.

---

## Install

```bash
pip install groundcheck[mcp]
```

This installs GroundCheck plus the `mcp` SDK. The `groundcheck-mcp` CLI is automatically available.

---

## Setup

### VS Code (GitHub Copilot)

Add to your workspace `.vscode/mcp.json` (or user `settings.json` under `"mcp"`):

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

### Claude Desktop

Add to `claude_desktop_config.json` (usually `%APPDATA%\Claude\` on Windows or `~/Library/Application Support/Claude/` on macOS):

```json
{
  "mcpServers": {
    "groundcheck": {
      "command": "groundcheck-mcp",
      "args": ["--db", ".groundcheck/memory.db", "--namespace", "my-project"]
    }
  }
}
```

### Cursor

Add to `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "groundcheck": {
      "command": "groundcheck-mcp",
      "args": ["--db", ".groundcheck/memory.db", "--namespace", "my-project"]
    }
  }
}
```

### Custom / Other MCP Clients

GroundCheck's MCP server communicates over **stdio**. Launch it as a subprocess and send/receive JSON-RPC over stdin/stdout:

```bash
groundcheck-mcp --db /path/to/memory.db --namespace my-project
```

---

## Namespaces — Project-Scoped Memory

Namespaces let you give each project its own memory scope while sharing personal facts globally.

| Namespace | Purpose | Example facts |
|-----------|---------|---------------|
| `"global"` | User-level facts shared across ALL projects | "User's name is Nick", "Prefers dark mode" |
| `"my-production-app"` | Project-specific conventions | "Enforce strict TypeScript", "All functions need docstrings" |
| `"playground"` | A personal/experimental project | "No docs needed", "Experimental — move fast" |

### How it works

1. Set `--namespace` when configuring the MCP server for each project
2. The AI stores **personal facts** with `namespace="global"` — these are visible everywhere
3. The AI stores **project facts** with the default namespace — these stay scoped to the project
4. When checking memory, both the project namespace AND `"global"` are queried by default

### Example workflow

```
# Production project — server configured with --namespace "prod-app"
User: "Always use strict TypeScript and document every function"
→ AI calls crt_store_fact("Always use strict TypeScript and document every function")
  → Stored in namespace "prod-app"

# Personal project — server configured with --namespace "sandbox"
User: "This is experimental, no docs needed"
→ AI calls crt_store_fact("This is experimental, no docs needed")
  → Stored in namespace "sandbox"

# In either project:
User: "My name is Nick"
→ AI calls crt_store_fact("User's name is Nick", namespace="global")
  → Visible in ALL projects
```

### Shared database, isolated memories

You can point multiple projects at the **same** `.groundcheck/memory.db` file — namespaces keep their memories separate while `"global"` facts flow everywhere:

```json
// Project A: .vscode/mcp.json
{"args": ["--db", "C:/Users/nick/.groundcheck/memory.db", "--namespace", "project-a"]}

// Project B: .vscode/mcp.json
{"args": ["--db", "C:/Users/nick/.groundcheck/memory.db", "--namespace", "project-b"]}
```

---

## Tools

The server exposes three tools:

### `crt_store_fact`

Store a user fact with automatic contradiction detection.

| Parameter   | Type   | Default     | Description |
|-------------|--------|-------------|-------------|
| `text`      | string | (required)  | The fact to store, e.g. "User works at Microsoft" |
| `source`    | string | `"user"`    | Source type: `user`, `document`, `code`, `inferred`. Affects trust score. |
| `thread_id` | string | `"default"` | Isolates memory per conversation/thread |
| `namespace` | string | `""`        | Project scope. Use `"global"` for personal facts. Empty = server's `--namespace` default. |

**Trust scores by source:**

| Source     | Trust |
|------------|-------|
| `user`     | 0.9   |
| `document` | 0.7   |
| `code`     | 0.8   |
| `inferred` | 0.4   |

**Returns:** JSON with `stored`, `memory_id`, `namespace`, `facts_extracted`, `contradictions`, `has_contradiction`.

**Example response:**
```json
{
  "stored": true,
  "memory_id": "a1b2c3",
  "text": "User works at Microsoft",
  "trust": 0.9,
  "facts_extracted": {"employer": "Microsoft"},
  "total_memories": 5,
  "contradictions": [],
  "has_contradiction": false
}
```

If a contradiction is detected (e.g. a previous memory says "works at Google"):
```json
{
  "stored": true,
  "contradictions": [
    {
      "slot": "employer",
      "values": ["microsoft", "google"],
      "most_trusted_value": "microsoft",
      "most_recent_value": "microsoft",
      "action": "Ask user to confirm which is current"
    }
  ],
  "has_contradiction": true
}
```

---

### `crt_check_memory`

Query stored facts before answering. Returns matching memories with trust scores and contradiction warnings.

| Parameter        | Type   | Default     | Description |
|------------------|--------|-------------|-------------|
| `query`          | string | (required)  | What to search for: "database", "employer", "location", etc. |
| `thread_id`      | string | `"default"` | Thread/conversation ID |
| `namespace`      | string | `""`        | Project scope to query. Empty = server's `--namespace` default. |
| `include_global` | bool   | `true`      | Also return `"global"` namespace memories (user-level facts) |

**Returns:** JSON with `found` count, `namespace`, `memories` array (each with `id`, `text`, `trust`, `timestamp`, `namespace`), and `contradictions` if any exist among returned memories.

**Example response:**
```json
{
  "found": 2,
  "memories": [
    {"id": "m1", "text": "User works at Microsoft", "trust": 0.9, "timestamp": 1707580000},
    {"id": "m2", "text": "User is a backend engineer", "trust": 0.9, "timestamp": 1707580100}
  ],
  "contradictions": []
}
```

---

### `crt_verify_output`

Verify a draft response against stored memories before sending it. Catches hallucinations and auto-corrects.

| Parameter        | Type   | Default     | Description |
|------------------|--------|-------------|-------------|
| `draft`          | string | (required)  | The draft text to verify |
| `thread_id`      | string | `"default"` | Thread/conversation ID |
| `mode`           | string | `"strict"`  | `strict` = rewrite hallucinations; `permissive` = report only |
| `namespace`      | string | `""`        | Project scope. Empty = server's `--namespace` default. |
| `include_global` | bool   | `true`      | Also verify against `"global"` namespace memories |

**Returns:** Full verification report with `passed`, `corrected`, `hallucinations`, `confidence`, `facts_extracted`, `contradiction_details`.

**Example — hallucination caught:**
```json
{
  "passed": false,
  "corrected": "You work at Microsoft and live in Seattle.",
  "hallucinations": ["Amazon"],
  "confidence": 0.65,
  "facts_extracted": {"employer": {"slot": "employer", "value": "Amazon", "normalized": "amazon"}},
  "contradicted_claims": [],
  "requires_disclosure": false
}
```

**Example — all grounded:**
```json
{
  "passed": true,
  "corrected": null,
  "hallucinations": [],
  "confidence": 0.95
}
```

---

## How It Works

```
User says something → Agent calls crt_store_fact → Fact stored with trust score
                                                  → Contradictions flagged

Agent needs context  → Agent calls crt_check_memory → Relevant memories returned
                                                    → Trust scores inform response

Agent writes reply   → Agent calls crt_verify_output → Hallucinations caught
                                                     → Corrections generated
                                                     → Confidence score calculated
```

All memories are persisted in SQLite (`--db` flag). Each thread gets isolated memory so concurrent conversations don't cross-contaminate.

---

## CLI Options

```
groundcheck-mcp [OPTIONS]

Options:
  --db PATH            SQLite database path (default: .groundcheck/memory.db)
  --namespace, -n NAME Project namespace for memory scoping (default: 'default')
  -v, --verbose        Enable debug logging
```

---

## Agent Instructions

When configuring your agent to use GroundCheck, include these instructions in the system prompt or MCP config:

> **Store facts** with `crt_store_fact` whenever the user states something about themselves, their project, preferences, or environment. Use `namespace="global"` for personal facts (name, preferences) that should be available in every project. Use the default namespace for project-specific facts (coding standards, tech stack, documentation rules).
>
> **Check memory** with `crt_check_memory` before answering questions about the user or their project. Global memories are included by default.
>
> **Verify output** with `crt_verify_output` before every response that references user facts. If verification fails, use the corrected text instead.

---

## Troubleshooting

**Server won't start:** Make sure `mcp` is installed: `pip install groundcheck[mcp]`

**No memories found:** Check you're using the same `thread_id` across store/check/verify calls.

**Database locked:** Only one MCP server instance should access each `.db` file at a time.

**Verbose mode:** Run with `-v` to see debug logs on stderr:
```bash
groundcheck-mcp --db .groundcheck/memory.db -v
```
