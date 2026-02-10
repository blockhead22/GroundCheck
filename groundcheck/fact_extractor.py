"""Lightweight fact-slot extraction for grounding verification.

Goal: reduce false contradiction triggers from generic semantic similarity by
comparing only facts that refer to the same attribute ("slot").

This is intentionally heuristic (no ML) and is tuned to the kinds of personal
profile facts used in conversational AI systems.
"""

from __future__ import annotations

import re
from typing import Dict

from .types import ExtractedFact


_WS_RE = re.compile(r"\s+")


def split_compound_values(text: str) -> list[str]:
    """Split compound values into individual claims.
    
    Handles multiple separators:
    - Commas: "Python, JavaScript, Ruby"
    - "and": "Python and JavaScript"
    - "or": "Python or JavaScript"
    - Slashes: "Python/JavaScript"
    - Semicolons: "Python; JavaScript"
    - Newlines/bullets: Multi-line lists
    - Mixed: "Python, JavaScript, and Ruby"
    
    Args:
        text: String that may contain compound values
        
    Returns:
        List of individual values
        
    Examples:
        >>> split_compound_values("Python, JavaScript, and Ruby")
        ['Python', 'JavaScript', 'Ruby']
        >>> split_compound_values("Python/JavaScript")
        ['Python', 'JavaScript']
        >>> split_compound_values("Python")
        ['Python']
    """
    if not text or not str(text).strip():
        return []
    
    text = str(text)
    
    # Handle newlines/bullets first (multi-line claims)
    if '\n' in text:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        # Recursively split each line
        result = []
        for line in lines:
            result.extend(split_compound_values(line))
        return result
    
    # Replace multiple separators with commas for uniform splitting
    # Order matters: process "and"/"or" before other separators
    normalized = text
    
    # Handle "X, Y, and Z" pattern (Oxford comma)
    normalized = re.sub(r',\s+and\s+', ', ', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r',\s+or\s+', ', ', normalized, flags=re.IGNORECASE)
    
    # Handle standalone "and"/"or"
    normalized = re.sub(r'\s+and\s+', ', ', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\s+or\s+', ', ', normalized, flags=re.IGNORECASE)
    
    # Handle slashes and semicolons
    normalized = normalized.replace('/', ', ')
    normalized = normalized.replace(';', ', ')
    
    # Handle bullets (•, -, *)
    normalized = re.sub(r'[•\-\*]\s*', '', normalized)
    
    # Split on commas and clean
    parts = [p.strip() for p in normalized.split(',')]
    
    # Filter out empty strings and common list artifacts
    cleaned = []
    for part in parts:
        part = part.strip()
        if part and part.lower() not in ['and', 'or', '&', 'the', 'a', 'an']:
            cleaned.append(part)
    
    return cleaned if cleaned else [text]


# Common company names to exclude from title extraction
_COMMON_COMPANY_NAMES = {'microsoft', 'google', 'amazon', 'apple', 'facebook', 'meta', 'netflix'}


_NAME_STOPWORDS = {
    # Common non-name tokens that appear after "I'm ..." in normal sentences.
    "a", "an", "the", "ai", "back", "building", "build", "busy", "fine",
    "good", "great", "here", "help", "okay", "ok", "ready", "sorry",
    "sure", "tired", "trying", "working", "going", "to",
}


def _norm_text(value: str) -> str:
    """Normalize text for comparison."""
    value = _WS_RE.sub(" ", value.strip())
    return value.lower()


def is_question(text: str) -> bool:
    """Check if text appears to be a question."""
    text = text.strip()
    if not text:
        return False
    if "?" in text:
        return True
    lowered = text.lower()
    return lowered.startswith((
        "what ", "where ", "when ", "why ", "how ", "who ", "which ",
        "do ", "does ", "did ", "can ", "could ", "should ", "would ",
        "is ", "are ", "am ", "was ", "were ", "tell me ",
    ))


def extract_fact_slots(text: str) -> Dict[str, ExtractedFact]:
    """Extract a small set of personal-profile fact slots from free text.
    
    Args:
        text: Input text to extract facts from
        
    Returns:
        Dictionary mapping slot names to ExtractedFact objects
    """
    facts: Dict[str, ExtractedFact] = {}

    if not text or not text.strip():
        return facts

    # Structured facts/preferences (useful for onboarding and explicit corrections).
    # Examples:
    # - "FACT: name = Nick"
    # - "PREF: communication_style = concise"
    structured = re.search(
        r"\b(?:FACT|PREF):\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+?)\s*$",
        text.strip(),
        flags=re.IGNORECASE,
    )
    if structured:
        slot = structured.group(1).strip().lower()
        value_raw = structured.group(2).strip()
        # Keep this conservative: whitelist the slots we intentionally support.
        allowed = {
            "name", "employer", "title", "location", "pronouns",
            "communication_style", "goals", "favorite_color",
        }
        if slot in allowed and value_raw:
            facts[slot] = ExtractedFact(slot, value_raw, _norm_text(value_raw))
            return facts

    # Name extraction patterns
    # Stop at coordinating conjunctions and punctuation  
    name_pat = r"([A-Za-z][A-Za-z'-]{1,40}(?:\s+[A-Za-z][A-Za-z'-]{1,40}){0,2})(?:(?:\s+and|\s+or|,|\.|;)|\s*$)"
    name_pat_title = r"([A-Z][A-Za-z'-]{1,40}(?:\s+[A-Z][A-Za-z'-]{1,40}){0,2})(?:(?:\s+and|\s+or|,|\.|;)|\s*$)"
    
    # Also extract from second and third person patterns
    # "Your name is X", "User's name is X"
    if "name" not in facts:
        m_other = re.search(r"\b(?:your|user'?s) name is\s+([A-Z][A-Za-z'-]{1,40}(?:\s+[A-Z][A-Za-z'-]{1,40}){0,2})(?:(?:\s+and|\s+or|,|\.|;)|\s*$)", text, flags=re.IGNORECASE)
        if m_other:
            name = m_other.group(1).strip()
            if name and name.lower() not in _NAME_STOPWORDS:
                facts["name"] = ExtractedFact("name", name, _norm_text(name))

    # Very explicit "call me" pattern
    if "name" not in facts:
        m = re.search(r"\bcall me\s+" + name_pat, text, flags=re.IGNORECASE)
    else:
        m = None
    if m:
        name = m.group(1).strip()
        tokens = [t for t in re.split(r"\s+", name) if t]
        token_lowers = [t.lower() for t in tokens]
        if tokens and not any(t in _NAME_STOPWORDS for t in token_lowers):
            facts["name"] = ExtractedFact("name", name, _norm_text(name))

    # Short correction pattern: "Nick not Ben"
    if "name" not in facts:
        m = re.match(
            r"^\s*([A-Z][A-Za-z'-]{1,40})\s+not\s+([A-Z][A-Za-z'-]{1,40})\s*[\.!?]?\s*$",
            text,
        )
        if m:
            cand = m.group(1).strip()
            if cand and cand.lower() not in _NAME_STOPWORDS:
                facts["name"] = ExtractedFact("name", cand, _norm_text(cand))

    # "My name is X" pattern
    m = re.search(r"\bmy name is\s+" + name_pat + r"\b", text, flags=re.IGNORECASE)
    if not m:
        # Prefer TitleCase names for the generic "I'm X" pattern
        m = re.search(r"\bi\s*['']?m\s+" + name_pat_title, text)
        if not m:
            # Also try "I am" pattern
            m = re.search(r"\bi\s+am\s+" + name_pat_title, text, flags=re.IGNORECASE)
        if not m:
            # Allow single-token lowercase name, but only when it appears as a direct name declaration
            m = re.search(r"^\s*i\s*['']?m\s+([a-z][a-z'-]{1,40})\s*[\.!?]?\s*$", text, flags=re.IGNORECASE)
    if m:
        name = m.group(1).strip()
        tokens = [t for t in re.split(r"\s+", name) if t]
        token_lowers = [t.lower() for t in tokens]

        # Filter obvious non-name phrases like "I'm trying to build ..."
        trailing = (text[m.end():] or "").lstrip().lower()
        looks_like_infinitive = trailing.startswith("to ")
        has_stopword = any(t in _NAME_STOPWORDS for t in token_lowers)

        if tokens and not has_stopword and not looks_like_infinitive:
            facts["name"] = ExtractedFact("name", name, _norm_text(name))

    # Compound introduction: "I am a Web Developer from Milwaukee Wisconsin"
    compound_intro = re.search(
        r"\bI (?:am|'m) (?:a |an )?(?P<occupation>[^,]+?)\s+(?:from|in)\s+(?P<location>.+?)(?:\.|$|,)",
        text,
        re.IGNORECASE
    )
    if compound_intro:
        occ = compound_intro.group("occupation").strip()
        loc = compound_intro.group("location").strip()
        # Only extract if occupation looks like a job title
        if occ and len(occ) > 2 and not any(word in occ.lower() for word in ["going", "coming", "person", "student", "happy", "sad"]):
            facts["occupation"] = ExtractedFact("occupation", occ, occ.lower())
        if loc and len(loc) > 2:
            facts["location"] = ExtractedFact("location", loc, loc.lower())

    # Employer extraction
    if re.search(r"\b(?:i work for myself|i'm self[- ]?employed|i am self[- ]?employed)", text, flags=re.IGNORECASE):
        facts["employer"] = ExtractedFact("employer", "self-employed", "self-employed")
    
    # "I run [business]" pattern
    m = re.search(r"\bi run (?:a |an )?([^\n\r\.;,]+?)(?:\s+(?:called|and|but|,|\.|;)|\s*$)", text, flags=re.IGNORECASE)
    if m and "employer" not in facts:
        business = m.group(1).strip()
        # Extract business name if "called X" follows
        m2 = re.search(r"called\s+([A-Z][A-Za-z0-9\s&\-\.]+?)(?:\s+(?:and|but|,|\.|;\()|\s*$)", text)
        if m2:
            business = m2.group(1).strip()
        if business:
            facts["employer"] = ExtractedFact("employer", f"self-employed ({business})", _norm_text(business))
    
    # "I work at/for X" pattern (first, second, and third person)
    if "employer" not in facts:
        # Primary pattern with subject pronoun
        m = re.search(
            r"\b(?:i|you|user|he|she|they) (?:currently )?(?:work(?:s)? (?:at|for)|(?:is|am|are) employed by)\s+([A-Z][A-Za-z0-9\s&\-\.]+?)(?:(?:\s+as|\s+and|\s+but|\s+in|\s+on|\s+for|\s+with|\s+where|\s*,|\.|;|\s+previously)|\s*$)",
            text,
            flags=re.IGNORECASE,
        )
        if not m:
            # Pattern for continuation after "and" (e.g., "lives in X and works at Y")
            m = re.search(r"\band\s+work(?:s)? (?:at|for)\s+([A-Z][A-Za-z0-9\s&\-\.]+?)(?:(?:\s+as|\s+and|\s+but|\s+in|,|\.|;)|\s*$)", text, flags=re.IGNORECASE)
        if not m:
            # Try "you're working at/for X" pattern
            m = re.search(r"\byou're working (?:at|for)\s+([A-Z][A-Za-z0-9\s&\-\.]+?)(?:(?:\s+as|\s+and|\s+but|,|\.|;)|\s*$)", text, flags=re.IGNORECASE)
        if not m:
            # Try "User is a [title] at [company]" pattern
            m = re.search(r"\b(?:user|he|she|they|i|you)\s+(?:is|am|are|was|were)\s+a\s+[A-Z][A-Za-z\s]+?\s+at\s+([A-Z][A-Za-z0-9\s&\-\.]+?)(?:\s+and|\s+in|,|\.|;|\s*$)", text, flags=re.IGNORECASE)
        if not m:
            # Try "[Name] is a [title] at [company]" pattern (for third-person references)
            m = re.search(r"\b[A-Z][a-z]+\s+(?:is|was)\s+a\s+[A-Z][A-Za-z\s]+?\s+at\s+([A-Z][A-Za-z0-9\s&\-\.]+?)(?:\s+and|\s+in|,|\.|;|\s*$)", text, flags=re.IGNORECASE)
        if m:
            employer_raw = m.group(1)
            # Trim at common continuations (redundant now but kept for safety)
            employer_raw = re.split(r"\b(?:as|and|but|in|though|however|previously)\b|[,\.;]", employer_raw, maxsplit=1, flags=re.IGNORECASE)[0]
            employer_raw = employer_raw.strip()
            if employer_raw:
                facts["employer"] = ExtractedFact("employer", employer_raw, _norm_text(employer_raw))

    # Job title / role / occupation
    # Try to extract from "as X" patterns first
    m = re.search(r"\bas\s+(?:a\s+)?([A-Z][A-Za-z\s]+?)(?:\s+(?:and|but|in|at|graduated)|\s*$)", text, flags=re.IGNORECASE)
    if m:
        title_raw = m.group(1).strip()
        # Avoid capturing company names as titles
        if len(title_raw.split()) <= 4 and title_raw.lower() not in _COMMON_COMPANY_NAMES:
            facts["title"] = ExtractedFact("title", title_raw, _norm_text(title_raw))
    
    if "title" not in facts:
        m = re.search(r"\bmy (?:role|job title|title) is\s+([^\n\r\.;,]+)", text, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"\b(?:i am a|i'm a)\s+([A-Z][A-Za-z\s]+?)(?:\s+(?:by|at|for|and)|\s*$)", text)
            if not m:
                # Try "User is a [title]" pattern
                m = re.search(r"\b(?:user|he|she|they)\s+(?:is|was)\s+a\s+([A-Z][A-Za-z\s]+?)(?:\s+(?:at|for|in|and|with)|\.|,|;|\s*$)", text, flags=re.IGNORECASE)
            if not m:
                m = re.search(r"\b([A-Z][A-Za-z\s]+?)\s+by\s+(?:degree|trade|profession)", text)
        if m:
            title_raw = m.group(1).strip()
            title_raw = re.split(r"\b(?:at|for|in|by)\b", title_raw, maxsplit=1, flags=re.IGNORECASE)[0].strip()
            if title_raw and len(title_raw.split()) <= 4:  # Keep it reasonable (1-4 words)
                facts["title"] = ExtractedFact("title", title_raw, _norm_text(title_raw))

    # Location (first, second, and third person)
    # Try explicit location patterns first
    m = re.search(r"\b(?:i|you|user|he|she|they) (?:lives?|resides?|moved to) in\s+(?:a\s+)?(?:\d+-bedroom\s+apartment\s+in\s+)?([A-Z][a-zA-Z .'-]+?)(?:\s+near|\s+with|\s+and|\.|,|;|\s*$)", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\b(?:i|you|user|he|she|they) moved to\s+([A-Z][a-zA-Z .'-]+?)(?:\s+near|\s+with|\s+and|\.|,|;|\s*$)", text, flags=re.IGNORECASE)
    if not m and "employer" in facts:
        # Check for "work at [company] in [location]" pattern
        m = re.search(r"\bworks? (?:at|for)\s+[A-Za-z0-9\s&\-\.]+?\s+in\s+([A-Z][a-zA-Z .'-]+?)(?:\s+near|\s+with|\s+and|\.|,|;|\s*$)", text, flags=re.IGNORECASE)
    if m:
        loc_value = m.group(1).strip()
        # Remove "in" prefix if present
        loc_value = re.sub(r'^\s*in\s+', '', loc_value, flags=re.IGNORECASE).strip()
        # Trim at common spatial modifiers (safety)
        loc_value = re.split(r'\s+(?:near|with)\s+', loc_value, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        # Split on temporal markers or punctuation and take first part
        loc_value = re.split(r"\s+(?:and|last|this|on|during)\s+|\.|,", loc_value, maxsplit=1)[0].strip()
        if loc_value:
            facts["location"] = ExtractedFact("location", loc_value, _norm_text(loc_value))

    # Years programming experience
    m = re.search(
        r"\b(?:i'?ve been programming for|i have been programming for)\s+(\d{1,3})\s+years\b",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        years = int(m.group(1))
        facts["programming_years"] = ExtractedFact("programming_years", years, str(years))

    # First programming language
    m = re.search(
        r"\b(?:starting with|started with|my first (?:programming )?language was)\s+([A-Z][A-Za-z0-9+_.#-]{1,40})\b",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        lang = m.group(1).strip()
        facts["first_language"] = ExtractedFact("first_language", lang, _norm_text(lang))

    # Team size
    m = re.search(r"\bteam of\s+(\d{1,3})\b", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bteam is\s+(\d{1,3})\b", text, flags=re.IGNORECASE)
    if m:
        size = int(m.group(1))
        facts["team_size"] = ExtractedFact("team_size", size, str(size))

    # Favorite color
    m = re.search(
        r"\bmy\s+favou?rite\s+colou?r\s+is\s+([^\n\r\.;,!\?]{2,60})",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        color_raw = m.group(1).strip()
        # Trim at common continuations
        color_raw = re.split(r"\b(?:and|but|though|however)\b", color_raw, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if color_raw:
            facts["favorite_color"] = ExtractedFact("favorite_color", color_raw, _norm_text(color_raw))

    # Additional facts from original implementation
    _extract_education_facts(text, facts)
    _extract_personal_facts(text, facts)
    _extract_professional_facts(text, facts)

    return facts


def _extract_education_facts(text: str, facts: Dict[str, ExtractedFact]) -> None:
    """Extract education-related facts."""
    # Combined pattern: "both my undergrad and Master's were from MIT"
    m = re.search(
        r"\bboth\s+my\s+(?:undergrad|undergraduate)(?:\s+degree)?\s+and\s+(?:my\s+)?master'?s(?:\s+degree)?\s+(?:were|was)?\s*(?:from|at)\s+([A-Z][A-Za-z .'-]{2,60})\b",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        school = m.group(1).strip()
        if school:
            facts["undergrad_school"] = ExtractedFact("undergrad_school", school, _norm_text(school))
            facts["masters_school"] = ExtractedFact("masters_school", school, _norm_text(school))

    m = re.search(r"\bundergraduate (?:degree )?was from\s+([A-Z][A-Za-z .'-]{2,60})\b", text, flags=re.IGNORECASE)
    if m:
        school = m.group(1).strip()
        facts["undergrad_school"] = ExtractedFact("undergrad_school", school, _norm_text(school))

    m = re.search(r"\bmaster'?s (?:degree )?.*?from\s+([A-Z][A-Za-z .'-]{2,60})\b", text, flags=re.IGNORECASE)
    if m:
        school = m.group(1).strip()
        facts["masters_school"] = ExtractedFact("masters_school", school, _norm_text(school))
    
    # Graduation year
    m = re.search(r"\b(?:i\s+)?graduated\s+(?:from.*)?(?:in|from.*in)\s+(19\d{2}|20\d{2})\b", text, flags=re.IGNORECASE)
    if m:
        year = m.group(1).strip()
        facts["graduation_year"] = ExtractedFact("graduation_year", year, year)
    
    # School (standalone "graduated from X" or "studied at X" patterns)
    if "school" not in facts:
        # Try "graduated from X" first
        m = re.search(r"\b(?:i\s+|you\s+)?graduated from\s+([A-Z][A-Za-z\s.'-]{1,50}?)(?:\s+in\s+\d{4}|\s+with|\.|,|;|\s+and|\s*$)", text, flags=re.IGNORECASE)
        if not m:
            # Try "studied at X" pattern.
            m = re.search(
                r"\b(?:i\s+|you\s+)?studied(?:\s+(?!at\b)[A-Za-z][A-Za-z\s]{0,40})?\s+at\s+([A-Z][A-Za-z\s.'-]{1,50}?)(?:\s+and|\.|,|;|\s*$)",
                text,
                flags=re.IGNORECASE,
            )
        if m:
            school = m.group(1).strip()
            facts["school"] = ExtractedFact("school", school, _norm_text(school))
    
    # Major/Degree field
    m = re.search(r"\b(?:degree|major)\s+in\s+([A-Z][A-Za-z\s]{2,40}?)(?:\s+from|\s+and|\s+with|\.|,|;|\s*$)", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(
            r"\bstudied\s+(?!at\b)([A-Z][A-Za-z\s]{2,40}?)(?:\s+at|\s*$)",
            text,
            flags=re.IGNORECASE,
        )
    if m:
        major = m.group(1).strip()
        # Filter out common false positives
        if major.lower().startswith("at "):
            major = major[3:].strip()
        if major.lower() not in ['university', 'college', 'school', 'institute'] and major:
            facts["major"] = ExtractedFact("major", major, _norm_text(major))
    
    # Minor
    m = re.search(r"\bminor\s+in\s+([A-Z][A-Za-z\s]{2,40}?)(?:\.|,|;|\s+and|\s*$)", text, flags=re.IGNORECASE)
    if m:
        minor = m.group(1).strip()
        facts["minor"] = ExtractedFact("minor", minor, _norm_text(minor))



def _extract_personal_facts(text: str, facts: Dict[str, ExtractedFact]) -> None:
    """Extract personal facts like hobbies, pets, preferences."""
    # Siblings
    m = re.search(r"\bi have\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+sibling", text, flags=re.IGNORECASE)
    if m:
        count_str = m.group(1).strip()
        word_to_num = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                       "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
        count_normalized = word_to_num.get(count_str.lower(), count_str)
        facts["siblings"] = ExtractedFact("siblings", count_normalized, count_normalized)
    
    # Languages spoken
    m = re.search(r"\bi speak\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+language", text, flags=re.IGNORECASE)
    if m:
        count_str = m.group(1).strip()
        word_to_num = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                       "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
        count_normalized = word_to_num.get(count_str.lower(), count_str)
        facts["languages_spoken"] = ExtractedFact("languages_spoken", count_normalized, count_normalized)
    
    # Pet (type and name)
    m = re.search(r"\bi have a\s+([a-z]+(?:\s+[a-z]+)?)\s+named\s+([A-Z][a-z]+)", text, flags=re.IGNORECASE)
    if m:
        pet_type = m.group(1).strip()
        pet_name = m.group(2).strip()
        facts["pet"] = ExtractedFact("pet", pet_type, _norm_text(pet_type))
        facts["pet_name"] = ExtractedFact("pet_name", pet_name, _norm_text(pet_name))
    else:
        m = re.search(r"\bmy (?:dog|cat|pet) is a\s+([a-z]+(?:\s+[a-z]+)?)", text, flags=re.IGNORECASE)
        # Note: Removed generic "[Name] is a [thing]" pattern that was matching
        # professional roles like "User is a Software Engineer". Only specific
        # pet-related patterns are used to avoid false positives.
    
    # Coffee preference
    m = re.search(r"\bi prefer\s+(dark|light|medium)\s+roast", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bmy coffee preference is\s+(dark|light|medium)\s+roast", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bswitched to\s+(dark|light|medium)\s+roast", text, flags=re.IGNORECASE)
    if m:
        coffee = m.group(1).strip() + " roast"
        facts["coffee"] = ExtractedFact("coffee", coffee, _norm_text(coffee))
    
    # Hobby (with compound value support)
    m = re.search(r"\bmy (?:weekend )?hobby is\s+([a-z][a-z\s-]{2,40}?)(?:\.|,|;|\s*$)", text, flags=re.IGNORECASE)
    if not m:
        # "you enjoy X and Y" or "you love X"
        # Capture hobby text up to terminators (handles compounds like "hiking and cooking")
        m = re.search(r"\b(?:you|user|i) (?:enjoy|love|like)(?:s)?\s+([a-z][a-z\s,\-]+?)(?:\s+and\s+you|\.|,\s+and\s+you|$)", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bi enjoy\s+([a-z][a-z\s-]{2,40}?)(?:\.|,|;|\s*$)", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\btaken up\s+([a-z][a-z\s-]{2,40}?)(?:\s+instead|\.|,|;|\s*$)", text, flags=re.IGNORECASE)
    if m:
        hobby = m.group(1).strip()
        facts["hobby"] = ExtractedFact("hobby", hobby, _norm_text(hobby))
    
    # Book currently reading
    m = re.search(r"\bi'?m reading ['\"]([^'\"]{5,80})['\"]", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bnow reading ['\"]([^'\"]{5,80})['\"]", text, flags=re.IGNORECASE)
    if m:
        book = m.group(1).strip()
        facts["book"] = ExtractedFact("book", book, _norm_text(book))
    
    # Children/family information
    # "with 2 kids", "have 3 children", "my son", "my daughter"
    m = re.search(r"\b(?:with|have|has)\s+(\d+|one|two|three|four|five)\s+(?:kid|child|children)s?", text, flags=re.IGNORECASE)
    if m:
        count_str = m.group(1).strip()
        word_to_num = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}
        count_normalized = word_to_num.get(count_str.lower(), count_str)
        facts["children"] = ExtractedFact("children", count_normalized, count_normalized)
    
    # Relationships: "my wife", "my husband", "married to"
    m = re.search(r"\b(?:my|married to|with my)\s+(wife|husband|partner|spouse)", text, flags=re.IGNORECASE)
    if m:
        rel_type = m.group(1).strip()
        facts["relationship"] = ExtractedFact("relationship", rel_type, _norm_text(rel_type))
    
    # Phone number
    m = re.search(r"\b(?:phone number|phone|cell|mobile)(?:\s+is|\s+:)?\s+([0-9\-\(\)\s]{7,20})", text, flags=re.IGNORECASE)
    if m:
        phone = m.group(1).strip()
        facts["phone"] = ExtractedFact("phone", phone, _norm_text(phone))
    
    # Email (already might be extracted elsewhere, but add for completeness)
    if "email" not in facts:
        m = re.search(r"\b(?:email|e-mail)(?:\s+is|\s+:)?\s+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", text, flags=re.IGNORECASE)
        if m:
            email = m.group(1).strip()
            facts["email"] = ExtractedFact("email", email, _norm_text(email))


def _extract_professional_facts(text: str, facts: Dict[str, ExtractedFact]) -> None:
    """Extract professional/work-related facts."""
    # Project name/description
    m = re.search(r"\bmy (?:current )?project\s+(?:is\s+called|'?s\s+name\s+is|name\s+is|is\s+building)\s+(?:a\s+)?([A-Za-z][A-Za-z0-9+_.#\s-]{1,60}?)(?:\.|,|;|\s+for|\s+that|\s+to|\s*$)", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bmy project focus\s+(?:has\s+)?shifted to\s+([A-Za-z][A-Za-z0-9+_.#\s-]{1,60}?)(?:\.|,|;|\s*$)", text, flags=re.IGNORECASE)
    if m:
        project = m.group(1).strip()
        facts["project"] = ExtractedFact("project", project, _norm_text(project))
    
    # Favorite programming language
    m = re.search(r"\bmy favorite (?:programming )?language is\s+([A-Z][A-Za-z0-9+#]{1,20})\b", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\b([A-Z][A-Za-z0-9+#]{1,20})\s+is (?:actually )?my favorite (?:programming )?language", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bi prefer\s+([A-Z][A-Za-z0-9+#]{1,20})\b", text, flags=re.IGNORECASE)
    if m:
        lang = m.group(1).strip()
        facts["programming_language"] = ExtractedFact("programming_language", lang, _norm_text(lang))
    
    # Programming languages/technologies from "use/know/works with" patterns (can be lists)
    # "You use Python, JavaScript, Ruby, and Go"
    # "User knows Python and JavaScript"
    if "programming_language" not in facts:
        m = re.search(
            r"\b(?:(?:i|you|user|he|she|they) (?:use|uses|know|knows|works? with)|user knows)\s+([A-Z][A-Za-z0-9+#,\s&-]+?)(?:\s*$|\.|\!)",
            text,
            flags=re.IGNORECASE
        )
        if m:
            lang_str = m.group(1).strip()
            # Store as single value for now - will be split by verifier if needed
            facts["programming_language"] = ExtractedFact("programming_language", lang_str, _norm_text(lang_str))
    
    # Employment history: "previously at X", "worked at X"
    m = re.search(r"\b(?:previously|formerly)\s+(?:at|worked at|employed by)\s+([A-Z][A-Za-z0-9\s&\-.]+?)(?:\s+and|\s+before|\.|,|;|\s*$)", text, flags=re.IGNORECASE)
    if m:
        prev_employer = m.group(1).strip()
        facts["previous_employer"] = ExtractedFact("previous_employer", prev_employer, _norm_text(prev_employer))
    
    # Job title hierarchy: "Senior X", "Lead X", "promoted to X"
    m = re.search(r"\bpromoted to\s+([A-Z][A-Za-z\s]{2,40}?)(?:\.|,|;|\s+at|\s*$)", text, flags=re.IGNORECASE)
    if m:
        promoted_title = m.group(1).strip()
        # This is the new title after promotion
        if "title" not in facts:
            facts["title"] = ExtractedFact("title", promoted_title, _norm_text(promoted_title))
    
    # Skills with proficiency levels
    # "expert in Python", "proficient in JavaScript"
    m = re.search(r"\b(?:expert|proficient|skilled|experienced)\s+(?:in|with)\s+([A-Z][A-Za-z0-9+#,\s&-]+?)(?:\s*$|\.|\!|,\s+and)", text, flags=re.IGNORECASE)
    if m:
        skill_str = m.group(1).strip()
        facts["skill"] = ExtractedFact("skill", skill_str, _norm_text(skill_str))
