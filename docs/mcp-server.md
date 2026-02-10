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
      "args": ["--db", ".groundcheck/memory.db"]
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
      "args": ["--db", ".groundcheck/memory.db"]
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
      "args": ["--db", ".groundcheck/memory.db"]
    }
  }
}
```

### Custom / Other MCP Clients

GroundCheck's MCP server communicates over **stdio**. Launch it as a subprocess and send/receive JSON-RPC over stdin/stdout:

```bash
groundcheck-mcp --db /path/to/memory.db
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

**Trust scores by source:**

| Source     | Trust |
|------------|-------|
| `user`     | 0.9   |
| `document` | 0.7   |
| `code`     | 0.8   |
| `inferred` | 0.4   |

**Returns:** JSON with `stored`, `memory_id`, `facts_extracted`, `contradictions`, `has_contradiction`.

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

| Parameter   | Type   | Default     | Description |
|-------------|--------|-------------|-------------|
| `query`     | string | (required)  | What to search for: "database", "employer", "location", etc. |
| `thread_id` | string | `"default"` | Thread/conversation ID |

**Returns:** JSON with `found` count, `memories` array (each with `id`, `text`, `trust`, `timestamp`), and `contradictions` if any exist among returned memories.

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

| Parameter   | Type   | Default     | Description |
|-------------|--------|-------------|-------------|
| `draft`     | string | (required)  | The draft text to verify |
| `thread_id` | string | `"default"` | Thread/conversation ID |
| `mode`      | string | `"strict"`  | `strict` = rewrite hallucinations; `permissive` = report only |

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
  --db PATH        SQLite database path (default: .groundcheck/memory.db)
  -v, --verbose    Enable debug logging
```

---

## Agent Instructions

When configuring your agent to use GroundCheck, include these instructions in the system prompt or MCP config:

> **Store facts** with `crt_store_fact` whenever the user states something about themselves, their project, preferences, or environment.
>
> **Check memory** with `crt_check_memory` before answering questions about the user or their project to ensure your response is grounded.
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
