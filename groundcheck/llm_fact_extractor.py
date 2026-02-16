"""LLM-based fact extraction — replaces the 1,200-line regex extractor.

Architecture:
    1. Structured FACT: pattern (5 lines, always runs first)
    2. Local LLM extraction (SmolLM-135M or any cached model)
    3. API extraction (GPT-4o-mini via GitHubModelsClient, dev-only)
    4. Regex fallback (legacy, for edge cases)

Config is loaded from:
    - Environment vars: GROUNDCHECK_EXTRACTOR_MODE, GROUNDCHECK_API_MODEL
    - .groundcheck_config.json (gitignored) for API settings

Usage:
    from groundcheck.llm_fact_extractor import extract_facts_llm

    facts = extract_facts_llm("I'm Nick, I live in Wisconsin")
    # -> {"name": ExtractedFact("name", "Nick", "nick"),
    #     "location": ExtractedFact("location", "Wisconsin", "wisconsin")}
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

from .types import ExtractedFact

logger = logging.getLogger("groundcheck.llm_extractor")

# ── Config ────────────────────────────────────────────────────────────────────

# Extraction mode: "llm" (default), "api", "hybrid", "regex" (legacy)
_EXTRACTOR_MODE = os.environ.get("GROUNDCHECK_EXTRACTOR_MODE", "llm")

# Local model for extraction (must be instruction-capable or we prompt-engineer)
_LOCAL_MODEL = os.environ.get("GROUNDCHECK_LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

# API model (only used when mode includes api)
_API_MODEL = os.environ.get("GROUNDCHECK_API_MODEL", "gpt-4o-mini")


# ── Cached model state ────────────────────────────────────────────────────────
_local_model = None
_local_tokenizer = None
_local_device = None
_api_client = None


# ── System prompt for extraction ──────────────────────────────────────────────
EXTRACTION_PROMPT = """Extract all personal facts from the text below. Return ONLY a JSON object mapping slot names to values. Use lowercase snake_case for slot names.

Common slots: name, location, occupation, employer, favorite_language, project, framework, editor, os, database, cloud, hobby, pet, pet_name, favorite_drink, favorite_food, favorite_color, school, major, certification, experience_years, team_size, salary, age, birthday, siblings, vehicle, side_project, pet_peeve, morning_routine, communication_style

Rules:
- Only extract facts that are explicitly stated
- If someone says "I'm not X", do NOT extract X as a fact
- Numbers should be plain digits (e.g., "8" not "eight")
- If no facts found, return {}
- Return ONLY the JSON, no explanation

Examples:
Text: "I'm Nick, a developer from Wisconsin"
{"name": "Nick", "occupation": "developer", "location": "Wisconsin"}

Text: "I use VS Code and write TypeScript mostly"
{"editor": "VS Code", "favorite_language": "TypeScript"}

Text: "My dog Luna is a golden retriever"
{"pet": "golden retriever", "pet_name": "Luna"}

Text: "What is the weather today?"
{}

Text: """


def _load_config() -> dict:
    """Load config from .groundcheck_config.json if it exists."""
    config_paths = [
        os.path.join(os.getcwd(), ".groundcheck_config.json"),
        os.path.join(os.path.expanduser("~"), ".groundcheck_config.json"),
    ]
    for path in config_paths:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
    return {}


def _get_local_model():
    """Lazy-load local model for extraction."""
    global _local_model, _local_tokenizer, _local_device

    if _local_model is not None:
        return _local_model, _local_tokenizer, _local_device

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_name = _LOCAL_MODEL
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info(f"Loading local extraction model: {model_name}")
        _local_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _local_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(device)
        _local_model.eval()
        _local_device = device

        # Set pad token if missing
        if _local_tokenizer.pad_token is None:
            _local_tokenizer.pad_token = _local_tokenizer.eos_token

        logger.info(f"Extraction model loaded on {device}")
        return _local_model, _local_tokenizer, _local_device

    except Exception as e:
        logger.warning(f"Failed to load local model: {e}")
        return None, None, None


def _get_api_client():
    """Lazy-load API client for extraction."""
    global _api_client

    if _api_client is not None:
        return _api_client

    try:
        # Try loading from config file first
        config = _load_config()
        token = config.get("github_token") or os.environ.get("GITHUB_TOKEN")

        if not token:
            # Try gh CLI
            import subprocess
            result = subprocess.run(
                ["gh", "auth", "token"],
                capture_output=True, text=True, timeout=10,
            )
            token = result.stdout.strip()

        if not token:
            logger.warning("No GitHub token available for API extraction")
            return None

        from openai import OpenAI
        _api_client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=token,
        )
        return _api_client

    except Exception as e:
        logger.warning(f"Failed to initialize API client: {e}")
        return None


# ── Extraction methods ────────────────────────────────────────────────────────

def _extract_structured(text: str) -> Dict[str, ExtractedFact]:
    """Fast path: extract FACT: slot = value patterns. Always runs first."""
    facts = {}
    m = re.search(
        r"\b(?:FACT|PREF):\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+?)\s*$",
        text.strip(),
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if m:
        slot = m.group(1).strip().lower()
        value = m.group(2).strip()
        if slot and value:
            facts[slot] = ExtractedFact(slot, value, value.lower(), source="structured")
    return facts


def _parse_llm_json(raw: str) -> dict:
    """Extract JSON from LLM output, handling markdown fences and noise."""
    # Try direct parse first
    raw = raw.strip()

    # Strip markdown code fences
    if "```" in raw:
        m = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()

    # Find first { ... } block
    brace_start = raw.find("{")
    if brace_start == -1:
        return {}

    depth = 0
    brace_end = -1
    for i in range(brace_start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                brace_end = i + 1
                break

    if brace_end == -1:
        return {}

    json_str = raw[brace_start:brace_end]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try fixing common JSON errors
        # Remove trailing commas
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)
        # Fix single quotes
        json_str = json_str.replace("'", '"')
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {}


def _dict_to_extracted_facts(d: dict, source: str = "llm") -> Dict[str, ExtractedFact]:
    """Convert a flat dict to ExtractedFact objects."""
    facts = {}
    for slot, value in d.items():
        if value is None or value == "":
            continue
        slot = str(slot).lower().strip().replace(" ", "_")
        value_str = str(value).strip()
        if value_str and slot:
            facts[slot] = ExtractedFact(slot, value_str, value_str.lower(), source=source)
    return facts


def _extract_with_local_model(text: str) -> Dict[str, ExtractedFact]:
    """Extract facts using local LLM."""
    model, tokenizer, device = _get_local_model()
    if model is None:
        return {}

    import torch

    prompt = EXTRACTION_PROMPT + f'"{text}"\n'

    # Check if model supports chat template (instruction-tuned models)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": "You extract personal facts from text and return JSON."},
            {"role": "user", "content": prompt},
        ]
        try:
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            input_text = prompt
    else:
        input_text = prompt

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens
    gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True)

    parsed = _parse_llm_json(raw)
    return _dict_to_extracted_facts(parsed, source="llm")


def _extract_with_api(text: str) -> Dict[str, ExtractedFact]:
    """Extract facts using GPT-4o-mini API."""
    client = _get_api_client()
    if client is None:
        return {}

    try:
        response = client.chat.completions.create(
            model=_API_MODEL,
            messages=[
                {"role": "system", "content": "You extract personal facts from text and return JSON only. No explanation."},
                {"role": "user", "content": EXTRACTION_PROMPT + f'"{text}"\n'},
            ],
            max_tokens=300,
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
        parsed = _parse_llm_json(raw)
        return _dict_to_extracted_facts(parsed, source="api")

    except Exception as e:
        logger.warning(f"API extraction failed: {e}")
        return {}


def _extract_with_regex(text: str) -> Dict[str, ExtractedFact]:
    """Fallback: use the legacy regex extractor."""
    try:
        from .fact_extractor import extract_fact_slots
        return extract_fact_slots(text)
    except Exception as e:
        logger.warning(f"Regex extraction fallback failed: {e}")
        return {}


# ── Main entry point ──────────────────────────────────────────────────────────

def extract_facts_llm(
    text: str,
    mode: Optional[str] = None,
    timeout_ms: int = 5000,
) -> Dict[str, ExtractedFact]:
    """Extract facts from text using LLM-based approach.

    Extraction pipeline:
        1. Structured FACT: patterns (always, <1ms)
        2. Based on mode:
           - "llm": Local model extraction
           - "api": GPT-4o-mini API extraction
           - "hybrid": Local model + API enhancement
           - "regex": Legacy regex fallback
        3. Merge results (structured always takes priority)

    Args:
        text: Input text to extract facts from
        mode: Override extraction mode (default: env var or "llm")
        timeout_ms: Max time for extraction in milliseconds

    Returns:
        Dict mapping slot names to ExtractedFact objects
    """
    if not text or not text.strip():
        return {}

    effective_mode = mode or _EXTRACTOR_MODE

    # Step 1: Always try structured FACT: patterns first
    facts = _extract_structured(text)
    if facts:
        # Structured patterns are unambiguous — return immediately
        return facts

    t0 = time.time()

    # Step 2: Mode-based extraction
    if effective_mode == "llm":
        facts = _extract_with_local_model(text)
        if not facts:
            # Fallback to regex if LLM produces nothing
            facts = _extract_with_regex(text)

    elif effective_mode == "api":
        facts = _extract_with_api(text)
        if not facts:
            facts = _extract_with_regex(text)

    elif effective_mode == "hybrid":
        # Try local first, enhance with API
        facts = _extract_with_local_model(text)
        elapsed = (time.time() - t0) * 1000
        if elapsed < timeout_ms:
            api_facts = _extract_with_api(text)
            # API supplements local — doesn't override
            for slot, fact in api_facts.items():
                if slot not in facts:
                    facts[slot] = fact
        if not facts:
            facts = _extract_with_regex(text)

    elif effective_mode == "regex":
        facts = _extract_with_regex(text)

    else:
        logger.warning(f"Unknown extractor mode: {effective_mode}, using regex")
        facts = _extract_with_regex(text)

    elapsed_ms = (time.time() - t0) * 1000
    if elapsed_ms > 1000:
        logger.info(f"Extraction took {elapsed_ms:.0f}ms ({effective_mode})")

    return facts


# ── Convenience: drop-in replacement for extract_fact_slots ───────────────────

def extract_fact_slots(text: str) -> Dict[str, ExtractedFact]:
    """Drop-in replacement for the legacy regex extract_fact_slots.

    Uses LLM extraction by default, with regex as fallback.
    Set GROUNDCHECK_EXTRACTOR_MODE=regex to use the old behavior.
    """
    return extract_facts_llm(text)
