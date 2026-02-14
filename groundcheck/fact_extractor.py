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
        # Accept any well-formed slot name (alphanumeric + underscores).
        # The structured FACT: format is an explicit declaration — trust it.
        if slot and value_raw:
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

    # Greeting pattern: "Hi Mike!", "Hey Sarah,", "Hello Dr. Jones"
    # Only match when the greeting is near the start of the text (first 50 chars)
    if "name" not in facts:
        m_greet = re.search(
            r"(?:^|\.\s+)(?:Hi|Hey|Hello|Yo|Howdy|Sup|Greetings)\s+([A-Z][A-Za-z'-]{1,40})(?:\s*[!,.\s]|$)",
            text[:80],
        )
        if m_greet:
            greet_name = m_greet.group(1).strip()
            if greet_name.lower() not in _NAME_STOPWORDS and greet_name.lower() not in {"there", "all", "everyone", "everybody", "folks", "team", "guys", "dear"}:
                facts["name"] = ExtractedFact("name", greet_name, _norm_text(greet_name))

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
        if not m:
            # Try "role/position/job/career/things at [company]" pattern
            # Catches: "your role at Disney", "things at PayPal", "my position at Google"
            m = re.search(r"\b(?:my|your|the|their|his|her)\s+(?:role|position|job|career|work|time|gig|stint|things)\s+at\s+([A-Z][A-Za-z0-9\s&\-\.]+?)(?:\s+and|\s+but|\s+in|\s+is|\s+was|\s+has|,|\.|;|\?|!|\s*$)", text, flags=re.IGNORECASE)
        if not m:
            # Try "[verb] at [company]" with contextual verbs
            # Catches: "started at Google", "joined Netflix", "left Amazon"
            m = re.search(r"\b(?:started|joined|left|quit|resigned from|hired at|employed at|interning at|interned at)\s+([A-Z][A-Za-z0-9\s&\-\.]+?)(?:\s+and|\s+but|\s+in|\s+as|,|\.|;|\s*$)", text, flags=re.IGNORECASE)
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
    if not m:
        # "life in X", "based in X", "living in X", "located in X"
        # Catches: "life in Miami", "how's life in Austin", "I'm based in NYC"
        m = re.search(r"\b(?:life|based|living|located|settling|settled)\s+in\s+([A-Z][a-zA-Z .'-]+?)(?:\s+near|\s+with|\s+and|\s+is|\s+has|\.|,|;|\?|!|\s*$)", text, flags=re.IGNORECASE)
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

    # General-purpose extraction — catches facts beyond the profile-specific
    # patterns above.  These run last so specific extractors take priority.
    _extract_age_and_date_facts(text, facts)
    _extract_quantitative_facts(text, facts)
    _extract_preference_and_opinion_facts(text, facts)
    _extract_technical_facts(text, facts)
    _extract_general_knowledge_facts(text, facts)

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


# ---------------------------------------------------------------------------
# General-purpose extraction functions
# ---------------------------------------------------------------------------

_WORD_TO_NUM = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
    "ninety": "90", "hundred": "100", "zero": "0",
}


def _extract_age_and_date_facts(text: str, facts: dict) -> None:
    """Extract age, birthday, and date-related facts."""

    # Age: "I'm 32", "I am 32 years old", "my age is 32", "age: 32"
    # Also second/third person: "You are 32", "User is 45 years old"
    if "age" not in facts:
        m = re.search(
            r"\b(?:i'?m|i am|you are|you're|he is|she is|they are|user is|my age is|age[:\s]+is?)\s+(\d{1,3})\s*(?:years?\s*old)?(?:\b|$)",
            text, flags=re.IGNORECASE,
        )
        if m:
            age = m.group(1)
            facts["age"] = ExtractedFact("age", age, age)

    # Birthday: "my birthday is March 15", "born on Jan 5 1990",
    # "DOB is 1990-01-15", "date of birth: March 15"
    if "birthday" not in facts:
        m = re.search(
            r"\b(?:my birthday is|born on|date of birth[:\s]+is?|dob[:\s]+is?)\s+"
            r"([A-Za-z0-9,\s/-]{4,30}?)(?:\.|;|\s+and|\s+in\s+[A-Z]|\s*$)",
            text, flags=re.IGNORECASE,
        )
        if m:
            bday = m.group(1).strip().rstrip(",")
            facts["birthday"] = ExtractedFact("birthday", bday, _norm_text(bday))

    # Birth year standalone: "I was born in 1992"
    if "birth_year" not in facts:
        m = re.search(r"\b(?:i was born|born)\s+in\s+(19\d{2}|20[0-2]\d)\b", text, flags=re.IGNORECASE)
        if m:
            year = m.group(1)
            facts["birth_year"] = ExtractedFact("birth_year", year, year)

    # Anniversary: "our anniversary is June 1", "married since 2015"
    if "anniversary" not in facts:
        m = re.search(
            r"\b(?:our anniversary is|anniversary[:\s]+is?|married since|married in)\s+"
            r"([A-Za-z0-9,\s/-]{3,30}?)(?:\.|;|\s*$)",
            text, flags=re.IGNORECASE,
        )
        if m:
            ann = m.group(1).strip().rstrip(",")
            facts["anniversary"] = ExtractedFact("anniversary", ann, _norm_text(ann))

    # Generic start/end dates: "started [the job/project/X] in YYYY"
    if "start_date" not in facts:
        m = re.search(
            r"\b(?:i |we )?(?:started|joined|began|commenced)\s+(?:the\s+)?(?:\w+\s+)?in\s+"
            r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|(?:19|20)\d{2})\b",
            text, flags=re.IGNORECASE,
        )
        if m:
            sd = m.group(1).strip()
            facts["start_date"] = ExtractedFact("start_date", sd, _norm_text(sd))

    if "end_date" not in facts:
        m = re.search(
            r"\b(?:deadline|due date|end date|expires?|expir(?:es|ation)|ends)\s+(?:is\s+|on\s+|:?\s*)"
            r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:,?\s*\d{4})?|"
            r"\d{4}[-/]\d{2}[-/]\d{2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b",
            text, flags=re.IGNORECASE,
        )
        if m:
            ed = m.group(1).strip()
            facts["end_date"] = ExtractedFact("end_date", ed, _norm_text(ed))

    # Duration: "been doing X for N years/months"
    if "duration" not in facts:
        m = re.search(
            r"\b(?:i'?ve been|been|i have been)\s+\w+(?:\s+\w+)?\s+for\s+"
            r"(\d{1,3})\s+(years?|months?|weeks?|days?)\b",
            text, flags=re.IGNORECASE,
        )
        if m:
            dur = f"{m.group(1)} {m.group(2)}"
            facts["duration"] = ExtractedFact("duration", dur, _norm_text(dur))


def _extract_quantitative_facts(text: str, facts: dict) -> None:
    """Extract general quantitative facts: salary, budget, measurements, counts."""

    # Salary / income: "$150k", "salary is $200,000", "I make $80k/year"
    if "salary" not in facts:
        m = re.search(
            r"\b(?:salary|income|pay|compensation|wage|i make|i earn)\s*(?:is|:|of)?\s*"
            r"[\$€£]?\s*(\d[\d,]*\.?\d*)\s*[kK]?(?:\s*(?:/\s*(?:year|yr|month|mo|hour|hr|annum))|"
            r"\s*(?:per|a)\s*(?:year|month|hour))?\b",
            text, flags=re.IGNORECASE,
        )
        if m:
            sal = m.group(0).strip()
            # Normalize: extract ust the number portion
            val = re.search(r"[\$€£]?\s*\d[\d,]*\.?\d*\s*[kK]?", sal)
            if val:
                facts["salary"] = ExtractedFact("salary", val.group(0).strip(), _norm_text(val.group(0).strip()))

    # Budget: "budget is $50,000"
    if "budget" not in facts:
        m = re.search(
            r"\bbudget\s*(?:is|:)\s*([\$€£]?\s*\d[\d,]*\.?\d*\s*[kKmMbB]?)\b",
            text, flags=re.IGNORECASE,
        )
        if m:
            budget = m.group(1).strip()
            facts["budget"] = ExtractedFact("budget", budget, _norm_text(budget))

    # Height: "5'11", "5 feet 11 inches", "180 cm", "height is X"
    if "height" not in facts:
        m = re.search(r"\b(?:height\s*(?:is|:)\s*|i'?m\s+)(\d{1,2}'\d{1,2}\"?|\d{1,3}\s*(?:cm|ft|feet|inches?))\b", text, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"\b(\d)'(\d{1,2})\"?\s*(?:tall)?\b", text)
        if m:
            height = m.group(0).strip()
            facts["height"] = ExtractedFact("height", height, _norm_text(height))

    # Weight: "180 lbs", "82 kg", "weigh 180"
    if "weight" not in facts:
        m = re.search(
            r"\b(?:weight\s*(?:is|:)\s*|i?\s*weigh\s+)(\d{2,3})\s*(lbs?|kg|kilos?|pounds?|stone)?\b",
            text, flags=re.IGNORECASE,
        )
        if m:
            weight = f"{m.group(1)} {m.group(2) or 'lbs'}".strip()
            facts["weight"] = ExtractedFact("weight", weight, _norm_text(weight))

    # Generalized count: "I have N Xs" (for things beyond siblings/children)
    # Catches: "I have 3 monitors", "I have two dogs", "we have 5 servers"
    if True:
        for m in re.finditer(
            r"\b(?:i|we)\s+have\s+(\d{1,4}|"
            + "|".join(_WORD_TO_NUM.keys())
            + r")\s+([a-z][a-z\s]{1,30}?)(?:\s+(?:and|but|in|on|at|that|which|running)|\.|,|;|\s*$)",
            text, flags=re.IGNORECASE,
        ):
            count_raw = m.group(1).strip()
            thing = m.group(2).strip()
            count_val = _WORD_TO_NUM.get(count_raw.lower(), count_raw)
            # Skip if already captured by specific extractors
            if thing.rstrip("s") in ("sibling", "child", "children", "kid", "language"):
                continue
            # Derive a reasonable slot name
            slot = re.sub(r"\s+", "_", thing.rstrip("s").strip())
            slot = re.sub(r"[^a-z0-9_]", "", slot.lower())
            if slot and slot not in facts:
                val_str = f"{count_val} {thing}"
                facts[slot] = ExtractedFact(slot, val_str, _norm_text(val_str))


def _extract_preference_and_opinion_facts(text: str, facts: dict) -> None:
    """Extract preferences, opinions, goals, and beliefs."""

    # General "my/your/user's favorite X is Y"
    # Handles: "my favorite color is blue", "your favorite food is pizza",
    # "User's favorite color is orange"
    for m in re.finditer(
        r"\b(?:my|your|user'?s?|his|her|their)\s+favou?rite\s+"
        r"([a-z][a-z\s]{0,20}?)\s+is\s+([^\n\r\.;,!\?]{2,60})",
        text, flags=re.IGNORECASE,
    ):
        subject = m.group(1).strip()
        value = m.group(2).strip()
        # Trim trailing conjunctions
        value = re.split(r"\b(?:and|but|though|however|because)\b", value, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        slot = "favorite_" + re.sub(r"\s+", "_", subject.lower())
        slot = re.sub(r"[^a-z0-9_]", "", slot)
        if slot not in facts and value:
            facts[slot] = ExtractedFact(slot, value, _norm_text(value))

    # "I like X" / "I love X" (for concrete things, not verbs)
    if "likes" not in facts:
        m = re.search(
            r"\bi (?:like|love|enjoy|am into|am a fan of)\s+"
            r"([^\n\r\.;!\?]{2,60}?)(?:\.|;|!|\s*$)",
            text, flags=re.IGNORECASE,
        )
        if m:
            val = m.group(1).strip()
            # Skip verb phrases ("I like to code") — keep noun phrases
            if not re.match(r"^to\s+", val, re.IGNORECASE):
                facts["likes"] = ExtractedFact("likes", val, _norm_text(val))

    # "I prefer X" / "I prefer X over Y"
    if "preference" not in facts:
        m = re.search(
            r"\bi prefer\s+([^\n\r\.;!\?]{2,60}?)(?:\s+over\s+([^\n\r\.;!\?]{2,60}))?"
            r"(?:\.|;|!|\s*$)",
            text, flags=re.IGNORECASE,
        )
        if m:
            preferred = m.group(1).strip()
            # Skip if already captured as coffee preference
            if "roast" not in preferred.lower():
                over = m.group(2)
                val = preferred + (f" over {over.strip()}" if over else "")
                facts["preference"] = ExtractedFact("preference", val, _norm_text(val))

    # Opinions: "I think X", "I believe X", "in my opinion X"
    if "opinion" not in facts:
        m = re.search(
            r"\b(?:i think|i believe|in my opinion|i feel that|my view is)\s+"
            r"([^\n\r\.;!\?]{5,120}?)(?:\.|;|!|\?|\s*$)",
            text, flags=re.IGNORECASE,
        )
        if m:
            opinion = m.group(1).strip()
            facts["opinion"] = ExtractedFact("opinion", opinion, _norm_text(opinion))

    # Goals / plans: "my goal is X", "I plan to X", "I'm trying to X"
    if "goal" not in facts:
        m = re.search(
            r"\b(?:my goal is|i(?:'m| am) (?:trying|planning|working|aiming) to|"
            r"i plan to|i want to|my plan is to|i aim to|working towards?)\s+"
            r"([^\n\r\.;!\?]{3,120}?)(?:\.|;|!|\s*$)",
            text, flags=re.IGNORECASE,
        )
        if m:
            goal = m.group(1).strip()
            facts["goal"] = ExtractedFact("goal", goal, _norm_text(goal))

    # Dislikes / avoidances: "I don't like X", "I hate X", "I avoid X"
    if "dislike" not in facts:
        m = re.search(
            r"\b(?:i (?:don'?t|do not) like|i hate|i avoid|i can'?t stand|"
            r"i'm allergic to|allergic to|i'?m intolerant to)\s+"
            r"([^\n\r\.;!\?]{2,80}?)(?:\.|;|!|\s*$)",
            text, flags=re.IGNORECASE,
        )
        if m:
            dislike = m.group(1).strip()
            facts["dislike"] = ExtractedFact("dislike", dislike, _norm_text(dislike))

    # Dietary restriction: "I'm vegan", "I'm vegetarian", "I eat halal", "I'm gluten-free"
    if "diet" not in facts:
        m = re.search(
            r"\b(?:i'?m|i am|i eat)\s+(vegan|vegetarian|pescatarian|keto|paleo|"
            r"halal|kosher|gluten[- ]?free|dairy[- ]?free|lactose[- ]?free)\b",
            text, flags=re.IGNORECASE,
        )
        if m:
            diet = m.group(1).strip()
            facts["diet"] = ExtractedFact("diet", diet, _norm_text(diet))


def _extract_technical_facts(text: str, facts: dict) -> None:
    """Extract technical/programming/infrastructure facts."""

    # Technology versions: "using Python 3.11", "Node 18", "React 18.2.0",
    # "running Java 21", "on Ruby 3.2"
    _tech_names = (
        r"(?:Python|Java|JavaScript|TypeScript|Node(?:\.?js)?|Ruby|Go|Rust|"
        r"C\+\+|C#|Swift|Kotlin|PHP|Perl|Scala|Elixir|Dart|R|Julia|"
        r"React|Angular|Vue|Svelte|Next\.?js|Django|Flask|FastAPI|"
        r"Spring\s?Boot|Rails|Laravel|Express|NestJS|"
        r"PostgreSQL|MySQL|MongoDB|Redis|SQLite|DynamoDB|"
        r"Docker|Kubernetes|Terraform|Ansible|"
        r"Ubuntu|Debian|CentOS|Fedora|macOS|Windows|Linux|"
        r"AWS|GCP|Azure|Vercel|Netlify|Heroku|"
        r"Nginx|Apache|Caddy|HAProxy|"
        r"Git|GitHub|GitLab|Bitbucket|"
        r"VS\s?Code|Vim|Neovim|Emacs|IntelliJ|PyCharm|WebStorm)"
    )

    # Versioned technology: "Python 3.11.4", "Node 18", "React 18.2"
    for m in re.finditer(
        r"\b" + _tech_names + r"\s+v?(\d+(?:\.\d+){0,3})\b",
        text, flags=re.IGNORECASE,
    ):
        tech = m.group(0).strip()
        # Normalize the tech name (strip version for slot name)
        tech_name = re.sub(r"\s+v?\d+(?:\.\d+){0,3}$", "", tech).strip()
        slot = re.sub(r"[\s.#+]+", "_", tech_name.lower()) + "_version"
        slot = re.sub(r"[^a-z0-9_]", "", slot)
        if slot not in facts:
            version = m.group(1) if m.lastindex and m.lastindex >= 1 else tech
            facts[slot] = ExtractedFact(slot, tech, _norm_text(tech))

    # Database: "our database is X", "using X as our db", "db is X"
    if "database" not in facts:
        m = re.search(
            r"\b(?:our )?(?:database|db)\s+(?:is|:)\s+([A-Z][A-Za-z0-9\s+#]{1,30}?)(?:\.|,|;|\s*$)",
            text, flags=re.IGNORECASE,
        )
        if not m:
            m = re.search(
                r"\busing\s+(PostgreSQL|MySQL|MongoDB|Redis|SQLite|"
                r"DynamoDB|Cassandra|CouchDB|Neo4j|MariaDB|Oracle|"
                r"SQL Server|Supabase|Firebase|ElasticSearch|ClickHouse)\b",
                text, flags=re.IGNORECASE,
            )
        if m:
            db = m.group(1).strip()
            facts["database"] = ExtractedFact("database", db, _norm_text(db))

    # Operating system: "running Ubuntu", "on macOS", "I use Windows 11"
    if "os" not in facts:
        m = re.search(
            r"\b(?:running|on|i use|using|my (?:os|operating system) is)\s+"
            r"(Ubuntu\s*\d*\.?\d*|Debian\s*\d*|CentOS\s*\d*|Fedora\s*\d*|"
            r"Arch(?:\s*Linux)?|macOS(?:\s*\w+)?|Windows\s*\d*|Linux\s*\w*)\b",
            text, flags=re.IGNORECASE,
        )
        if m:
            os_val = m.group(1).strip()
            facts["os"] = ExtractedFact("os", os_val, _norm_text(os_val))

    # Editor / IDE: "my editor is X", "I use VS Code"
    if "editor" not in facts:
        m = re.search(
            r"\b(?:my (?:editor|ide) is|i (?:use|prefer))\s+"
            r"(VS\s?Code|Visual Studio(?:\s+Code)?|Vim|Neovim|Emacs|"
            r"IntelliJ(?:\s+IDEA)?|PyCharm|WebStorm|Sublime(?:\s+Text)?|"
            r"Atom|Cursor|Zed|Helix|Nano)\b",
            text, flags=re.IGNORECASE,
        )
        if m:
            editor = m.group(1).strip()
            facts["editor"] = ExtractedFact("editor", editor, _norm_text(editor))

    # Framework / stack: "built with React", "our stack is X", "using Django"
    if "framework" not in facts:
        m = re.search(
            r"\b(?:built with|framework is|stack is|using)\s+"
            r"(React|Angular|Vue(?:\.?js)?|Svelte|Next\.?js|Nuxt|Remix|Astro|"
            r"Django|Flask|FastAPI|Express(?:\.?js)?|NestJS|Rails|Laravel|"
            r"Spring\s?Boot|ASP\.NET|Phoenix|Gin|Fiber|Actix|Rocket)\b",
            text, flags=re.IGNORECASE,
        )
        if m:
            fw = m.group(1).strip()
            facts["framework"] = ExtractedFact("framework", fw, _norm_text(fw))

    # Cloud provider: "deployed on AWS", "hosted on GCP", "running on Azure"
    if "cloud" not in facts:
        m = re.search(
            r"\b(?:deployed on|hosted on|running on|using|on)\s+"
            r"(AWS|GCP|Google\s+Cloud|Azure|Vercel|Netlify|Heroku|"
            r"DigitalOcean|Linode|Fly\.io|Railway|Render)\b",
            text, flags=re.IGNORECASE,
        )
        if m:
            cloud = m.group(1).strip()
            facts["cloud"] = ExtractedFact("cloud", cloud, _norm_text(cloud))

    # Configuration values: "port is 8080", "timeout is 30s", "max retries is 3"
    for m in re.finditer(
        r"\b(port|timeout|max[_\s]?retries|rate[_\s]?limit|"
        r"batch[_\s]?size|workers?|threads?|ttl|interval|threshold|"
        r"concurrency|buffer[_\s]?size|max[_\s]?connections)\s+"
        r"(?:is|=|:)\s*(\d[\d.,]*\s*(?:s|ms|sec|seconds?|min|minutes?|hrs?|hours?|mb|gb|kb)?)\b",
        text, flags=re.IGNORECASE,
    ):
        config_key = re.sub(r"\s+", "_", m.group(1).strip().lower())
        config_val = m.group(2).strip()
        if config_key not in facts:
            facts[config_key] = ExtractedFact(config_key, config_val, _norm_text(config_val))

    # API URL / endpoint: "the API is at X", "endpoint is X", "API URL: X"
    if "api_url" not in facts:
        m = re.search(
            r"\b(?:api|endpoint|url|base[_\s]?url|server)\s+(?:is\s+(?:at\s+)?|(?:url\s+)?(?:is|:)\s*|at\s+)"
            r"(https?://[^\s,;\"'<>]{5,120})",
            text, flags=re.IGNORECASE,
        )
        if m:
            url = m.group(1).strip()
            facts["api_url"] = ExtractedFact("api_url", url, _norm_text(url))

    # Conversational coding patterns:
    # "I code in Python", "I usually code in TypeScript", "I program in Rust",
    # "I write Python", "I mostly write Go"
    _lang_list = (
        r"Python|Java|JavaScript|TypeScript|Ruby|Go|Rust|"
        r"C\+\+|C#|Swift|Kotlin|PHP|Perl|Scala|Elixir|Dart|Julia|"
        r"Lua|Haskell|Clojure|F#|OCaml|Zig|Carbon|Mojo"
    )
    if "programming_language" not in facts:
        m = re.search(
            r"\bi\s+(?:usually\s+|mostly\s+|primarily\s+|mainly\s+)?"
            r"(?:code|program|develop|write)\s+(?:in\s+)?"
            r"(" + _lang_list + r")"
            r"(?:\s+and\s+(" + _lang_list + r"))?",
            text, flags=re.IGNORECASE,
        )
        if m:
            lang = m.group(1).strip()
            lang2 = m.group(2).strip() if m.group(2) else None
            val = lang + (f" and {lang2}" if lang2 else "")
            facts["programming_language"] = ExtractedFact(
                "programming_language", val, _norm_text(val),
            )

    # Communication / coding style: "I prefer concise code",
    # "I like detailed comments", "keep it simple"
    if "coding_style" not in facts:
        m = re.search(
            r"\b(?:i (?:like|prefer|want)|keep it|use)\s+"
            r"(concise|verbose|detailed|minimal|simple|clean|dry|"
            r"functional|object[- ]?oriented|OOP|readable|pragmatic|"
            r"strict|loose|explicit|implicit)\s*(?:code|style|approach)?",
            text, flags=re.IGNORECASE,
        )
        if m:
            style = m.group(1).strip()
            facts["coding_style"] = ExtractedFact(
                "coding_style", style, _norm_text(style),
            )

    # Documentation preference: "I need docs", "no docs needed",
    # "always document", "skip documentation"
    if "docs_preference" not in facts:
        m = re.search(
            r"\b(?:always\s+(?:write|add|include)\s+(?:docs|documentation|docstrings)|"
            r"(?:no|skip|don'?t need|don'?t want)\s+(?:docs|documentation|docstrings)|"
            r"(?:docs|documentation)\s+(?:required|needed|not needed|optional|mandatory)|"
            r"every\s+(?:function|method|class)\s+(?:needs?|should have)\s+(?:a\s+)?(?:docs?tring|documentation))",
            text, flags=re.IGNORECASE,
        )
        if m:
            pref = m.group(0).strip()
            facts["docs_preference"] = ExtractedFact(
                "docs_preference", pref, _norm_text(pref),
            )

    # Testing preference: "I use pytest", "we use jest", "prefer unit tests"
    if "testing" not in facts:
        m = re.search(
            r"\b(?:i use|we use|prefer|using)\s+"
            r"(pytest|jest|mocha|vitest|cypress|playwright|selenium|"
            r"unittest|rspec|minitest|junit|xunit|nunit|go test)\b",
            text, flags=re.IGNORECASE,
        )
        if m:
            test_tool = m.group(1).strip()
            facts["testing"] = ExtractedFact(
                "testing", test_tool, _norm_text(test_tool),
            )


def _extract_general_knowledge_facts(text: str, facts: dict) -> None:
    """Universal catch-all extraction for declarative claims.

    This runs last and only fires for slots not already claimed by the
    more specific extractors above.  It handles:

    1. "my/the/our X is Y" (possessive/article + copular verb)
    2. "X is Y" (bare subject + copular verb)
    3. "X uses/handles/supports/runs Y" (subject + action verb)
    4. "X requires/needs/must have Y" (requirements/constraints)
    5. "We agreed/decided to X" (decisions)
    6. "X should be/must be/needs to be Y" (prescriptive config)
    7. "X is set to Y" / "X equals Y" (configuration)

    Clause splitting: sentences with commas/semicolons are split and
    each clause is re-parsed to avoid losing secondary facts.
    """

    # Blocklist: subjects that are too generic or cause false positives
    _SUBJECT_BLOCKLIST = {
        "thing", "stuff", "problem", "issue", "point", "question", "answer",
        "fact", "truth", "reason", "way", "idea",
        "it", "this", "that", "he", "she", "they", "we", "you",
        "name", "age", "job", "role",  # Already handled by specific extractors
        # Question words (prevent false extraction from interrogative sentences)
        "how", "what", "where", "when", "why", "who", "which",
    }

    def _try_store(subject: str, value: str) -> None:
        """Normalize and store a subject-value pair if it's valid."""
        # Strip leading possessive/article from subject before normalizing
        subject = re.sub(r"^(?:my|your|our|his|her|their|the)\s+", "", subject, flags=re.IGNORECASE)
        # Normalize slot name
        slot = re.sub(r"['\s]+", "_", subject.lower()).strip("_")
        slot = re.sub(r"[^a-z0-9_]", "", slot)

        if not slot or not value or len(slot) < 2:
            return
        if slot in facts:
            return
        if slot in _SUBJECT_BLOCKLIST:
            return
        # Reject if value starts with a common continuation word (likely not a fact)
        if re.match(r"^(?:that|not|also|just|still|always|never|really|very)\b", value, re.IGNORECASE):
            return
        if len(value.strip()) < 1:
            return

        # Trim trailing conjunctions
        value = re.split(r"\b(?:and|but|so|though|because|however|which)\b", value, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if value:
            facts[slot] = ExtractedFact(slot, value, _norm_text(value))

    # ── Shared regex fragments ─────────────────────────────────────
    # Value capture: any char except newline/CR/semicolon/exclamation/question
    # Periods are allowed WITHIN values (e.g. "99.9%", "v3.11", "api.example.com")
    _VAL = r"[^\n\r;!\?]"
    # Sentence terminator: period NOT preceded by digit, or ;!? or end-of-string
    _END = r"(?:(?<!\d)\.|;|!|\?|\s*$)"

    # ── Clause splitting ─────────────────────────────────────────────
    # Split on commas and semicolons to handle compound sentences like
    # "The frontend is React, backend is FastAPI"
    clauses = re.split(r"[;]|\s*,\s+(?=[a-z])", text, flags=re.IGNORECASE)

    for clause in clauses:
        clause = clause.strip()
        if not clause or len(clause) < 5:
            continue

        # ── Pattern 1: "[article/possessive] X is/are/was/were Y" ────
        for m in re.finditer(
            r"\b(?:my|the|our|his|her|their)\s+"
            r"([a-z][a-z\s']{0,30}?)\s+(?:is|are|was|were)\s+"
            rf"({_VAL}{{1,80}}?){_END}",
            clause, flags=re.IGNORECASE,
        ):
            _try_store(m.group(1).strip(), m.group(2).strip())

        # ── Pattern 2: "X is/are Y" (bare subject, no article needed) ──
        # Accepts both capitalized starts and lowercase after clause split
        for m in re.finditer(
            r"(?:^|\.\s+)"
            r"([A-Za-z][a-z]+(?:\s+[a-z]+){0,2})\s+(?:is|are|was|were)\s+"
            rf"({_VAL}{{1,80}}?){_END}",
            clause, flags=re.IGNORECASE,
        ):
            _try_store(m.group(1).strip(), m.group(2).strip())

        # ── Pattern 3: "X uses/handles/supports/runs/provides Y" ──────
        for m in re.finditer(
            r"\b(?:the|our|my|their)?\s*"
            r"([a-z][a-z\s']{0,30}?)\s+"
            r"(?:uses?|handles?|supports?|runs?|provides?|utilizes?|leverages?|relies on|is powered by|is built (?:with|on|using))\s+"
            rf"({_VAL}{{1,80}}?){_END}",
            clause, flags=re.IGNORECASE,
        ):
            _try_store(m.group(1).strip(), m.group(2).strip())

        # ── Pattern 4: "X requires/needs/demands/mandates Y" ──────────
        for m in re.finditer(
            r"\b(?:the|our|my|their)?\s*"
            r"([a-z][a-z\s']{0,30}?)\s+"
            r"(?:requires?|needs?|demands?|mandates?|expects?)\s+"
            rf"({_VAL}{{1,80}}?){_END}",
            clause, flags=re.IGNORECASE,
        ):
            _try_store(m.group(1).strip(), m.group(2).strip())

        # ── Pattern 5: "We agreed/decided to X" / "We chose X" ────────
        m = re.search(
            r"\b(?:we|they|the team|I)\s+"
            r"(?:agreed|decided|chose|committed|opted)\s+"
            r"(?:to\s+)?(?:use\s+|go with\s+|adopt\s+|implement\s+|switch to\s+)?"
            rf"({_VAL}{{1,80}}?){_END}",
            clause, flags=re.IGNORECASE,
        )
        if m:
            value = m.group(1).strip()
            # Try to infer a slot name from the context
            if re.search(r"REST|GraphQL|SOAP|gRPC", value, re.IGNORECASE):
                _try_store("api_style", value)
            elif re.search(r"arch|pattern|micro|mono", value, re.IGNORECASE):
                _try_store("architecture", value)
            else:
                _try_store("decision", value)

        # ── Pattern 6: "X should be / must be / needs to be Y" ────────
        for m in re.finditer(
            r"\b(?:the|our|my|their)?\s*"
            r"([a-z][a-z_\s]{1,25}?)\s+"
            r"(?:should\s+be|must\s+be|needs?\s+to\s+be|has\s+to\s+be|ought\s+to\s+be)\s+"
            rf"({_VAL}{{1,60}}?){_END}",
            clause, flags=re.IGNORECASE,
        ):
            _try_store(m.group(1).strip(), m.group(2).strip())

        # ── Pattern 7: "X is set to Y" / "X is configured as Y" ──────
        for m in re.finditer(
            r"\b([a-z][a-z_\s]{1,25}?)\s+is\s+(?:set to|configured (?:as|to)|currently)\s+"
            rf"({_VAL}{{1,60}}?){_END}",
            clause, flags=re.IGNORECASE,
        ):
            _try_store(m.group(1).strip(), m.group(2).strip())

        # ── Pattern 8: "X is handled/managed/done via/by/through Y" ──
        for m in re.finditer(
            r"\b([a-z][a-z\s']{0,30}?)\s+"
            r"is\s+(?:handled|managed|done|performed|implemented|achieved|provided)\s+"
            r"(?:via|by|through|using|with)\s+"
            rf"({_VAL}{{1,80}}?){_END}",
            clause, flags=re.IGNORECASE,
        ):
            _try_store(m.group(1).strip(), m.group(2).strip())

        # ── Pattern 9: "X equals Y" / "X = Y" ────────────────────────
        for m in re.finditer(
            r"\b([a-z][a-z_\s]{1,25}?)\s+(?:equals?|==?)\s+"
            rf"({_VAL}{{1,60}}?){_END}",
            clause, flags=re.IGNORECASE,
        ):
            _try_store(m.group(1).strip(), m.group(2).strip())

