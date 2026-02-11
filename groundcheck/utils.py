"""Utility functions for GroundCheck library."""

from __future__ import annotations

import re
from typing import Set


def normalize_text(text: str) -> str:
    """Normalize text for comparison by lowercasing and collapsing whitespace.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def extract_memory_claim_phrases() -> Set[str]:
    """Get set of phrases that indicate memory claims.
    
    Returns:
        Set of memory claim phrases
    """
    return {
        "i remember",
        "i recall",
        "i have a memory",
        "i have it noted",
        "i have you down",
        "i have stored",
        "in my memory",
        "in my notes",
        "i've got it stored",
        "i've got you stored",
        "i've got it noted",
        "i've got you down",
    }


def has_memory_claim(text: str) -> bool:
    """Check if text contains memory claim phrases.
    
    Args:
        text: Text to check
        
    Returns:
        True if text contains memory claim phrases
    """
    if not text or not text.strip():
        return False
    
    text_lower = text.lower()
    memory_phrases = extract_memory_claim_phrases()
    return any(phrase in text_lower for phrase in memory_phrases)


def create_memory_claim_regex() -> re.Pattern:
    """Create regex pattern for detecting memory claim lines.
    
    Returns:
        Compiled regex pattern
    """
    return re.compile(
        r"\b(i\s+(remember|recall)|i\s+have\s+(a\s+)?memory|i\s+have\s+it\s+noted|"
        r"i\s+have\s+you\s+down|i\s+have\s+stored|in\s+my\s+(memory|notes)|"
        r"i'?ve\s+got\s+(it|you)\s+(stored|noted|down))\b",
        re.I,
    )


def parse_fact_from_memory_text(text: str) -> tuple[str, str] | None:
    """Parse structured FACT: slot = value from memory text.
    
    Args:
        text: Memory text to parse
        
    Returns:
        Tuple of (slot, value) if match found, None otherwise
    """
    fact_re = re.compile(r"^\s*fact:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+?)\s*$", re.I)
    m = fact_re.match(text)
    if not m:
        return None
    
    slot = m.group(1).strip().lower()
    value = m.group(2).strip()
    
    if not slot or not value:
        return None
    
    return (slot, value)
