"""GroundCheck Extraction Benchmark Suite.

Measures precision, recall, and F1 for three extraction tiers:
    - Tier 1: Regex only (extract_fact_slots)
    - Tier 1.5: Knowledge only (extract_knowledge_facts)
    - Combined: Regex + Knowledge (as wired in verifier)

Usage:
    python -m benchmarks.run                  # Full report
    python -m benchmarks.run --style conv     # Only conversational sentences
    python -m benchmarks.run --json           # Machine-readable output
    python -m benchmarks.run --verbose        # Show per-sentence detail
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# Add parent to path so we can import groundcheck
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groundcheck.fact_extractor import extract_fact_slots
from groundcheck.knowledge_extractor import extract_knowledge_facts
from groundcheck.types import ExtractedFact


# ── Data loading ──────────────────────────────────────────────────────────────

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BENCHMARK_DIR, "dataset.json")


def load_dataset(
    style_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    difficulty_filter: Optional[str] = None,
) -> List[dict]:
    """Load benchmark dataset, optionally filtered."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter out section markers
    items = [d for d in data if "id" in d]

    if style_filter:
        items = [d for d in items if d.get("style") == style_filter]
    if category_filter:
        items = [d for d in items if d.get("category") == category_filter]
    if difficulty_filter:
        items = [d for d in items if d.get("difficulty") == difficulty_filter]

    return items


# ── Extraction runners ────────────────────────────────────────────────────────

# Slot name normalization: regex and knowledge use different names
# for the same concept. Map to a common namespace for fair comparison.
SLOT_ALIASES: Dict[str, str] = {
    # Regex slot → canonical
    "programming_language": "language",
    "framework": "backend_framework",
    "cloud": "cloud_provider",
    # Knowledge slots already use canonical names
}


def _normalize_slot(slot: str) -> str:
    """Normalize a slot name to canonical form."""
    return SLOT_ALIASES.get(slot, slot)


def _normalize_value(value: str) -> str:
    """Normalize a value for comparison (lowercase, strip)."""
    return value.strip().lower()


# Known aliases for entity values (PostgreSQL == Postgres, etc.)
VALUE_ALIASES: Dict[str, str] = {
    "postgres": "postgresql",
    "mongo": "mongodb",
    "k8s": "kubernetes",
    "gcp": "google cloud platform",
    "google cloud": "google cloud platform",
    "next.js": "nextjs",
    "nextjs": "next.js",
    "vue.js": "vuejs",
    "react.js": "reactjs",
    "node": "node.js",
    "ruby on rails": "rails",
    "github actions": "github actions",
    "tailwind": "tailwind css",
}


def _normalize_for_match(value: str) -> str:
    """Normalize value for matching, resolving common aliases."""
    v = _normalize_value(value)
    return VALUE_ALIASES.get(v, v)


def _values_match(extracted: str, expected: str) -> bool:
    """Check if an extracted value matches an expected value (fuzzy)."""
    e1 = _normalize_for_match(extracted)
    e2 = _normalize_for_match(expected)
    # Exact match
    if e1 == e2:
        return True
    # One contains the other
    if e1 in e2 or e2 in e1:
        return True
    return False


def run_regex_only(text: str) -> Dict[str, str]:
    """Run regex extraction and return normalized slot→value dict."""
    raw = extract_fact_slots(text)
    return {_normalize_slot(k): v.value for k, v in raw.items()}


def run_knowledge_only(text: str) -> Dict[str, str]:
    """Run knowledge extraction and return normalized slot→value dict."""
    raw = extract_knowledge_facts(text)
    return {_normalize_slot(k): v.value for k, v in raw.items()}


def run_combined(text: str) -> Dict[str, str]:
    """Run combined extraction (regex + knowledge, knowledge fills gaps)."""
    regex_raw = extract_fact_slots(text)
    knowledge_raw = extract_knowledge_facts(text)

    result = {_normalize_slot(k): v.value for k, v in regex_raw.items()}
    covered = set(result.keys())
    # Add known regex aliases to covered set
    for orig_slot in regex_raw:
        canonical = _normalize_slot(orig_slot)
        covered.add(canonical)
        covered.add(orig_slot)

    for k, v in knowledge_raw.items():
        canonical = _normalize_slot(k)
        if canonical not in covered:
            result[canonical] = v.value

    return result


# ── Scoring ───────────────────────────────────────────────────────────────────

@dataclass
class SlotScore:
    """Score for a single slot extraction attempt."""
    slot: str
    expected_value: str
    extracted_value: Optional[str]
    correct: bool


@dataclass
class SentenceResult:
    """Result of benchmarking one sentence across one tier."""
    id: str
    text: str
    expected: Dict[str, str]
    extracted: Dict[str, str]
    true_positives: int = 0   # Correctly extracted slots
    false_positives: int = 0  # Extracted but not expected
    false_negatives: int = 0  # Expected but not extracted
    slot_scores: List[SlotScore] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class TierMetrics:
    """Aggregate metrics for one extraction tier."""
    tier_name: str
    total_sentences: int = 0
    total_expected_slots: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_latency_ms: float = 0.0
    sentence_results: List[SentenceResult] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_sentences if self.total_sentences > 0 else 0.0


def score_sentence(
    item: dict,
    extractor_fn,
    tier_name: str,
) -> SentenceResult:
    """Score a single benchmark item against an extractor."""
    text = item["text"]
    expected = item["expected"]

    # Time the extraction
    start = time.perf_counter()
    extracted = extractor_fn(text)
    elapsed_ms = (time.perf_counter() - start) * 1000

    result = SentenceResult(
        id=item["id"],
        text=text,
        expected=expected,
        extracted=extracted,
        latency_ms=elapsed_ms,
    )

    # Normalize expected slots
    expected_norm = {_normalize_slot(k): v for k, v in expected.items()}
    extracted_norm = {_normalize_slot(k): v for k, v in extracted.items()}

    # True positives: expected slots that were correctly extracted
    for slot, expected_val in expected_norm.items():
        if slot in extracted_norm and _values_match(extracted_norm[slot], expected_val):
            result.true_positives += 1
            result.slot_scores.append(SlotScore(slot, expected_val, extracted_norm[slot], True))
        else:
            result.false_negatives += 1
            result.slot_scores.append(SlotScore(
                slot, expected_val,
                extracted_norm.get(slot), False
            ))

    # False positives: extracted slots not in expected
    # Only count if the sentence has expected facts (skip-category excluded)
    if expected_norm:
        for slot in extracted_norm:
            if slot not in expected_norm:
                result.false_positives += 1

    return result


def run_benchmark(
    dataset: List[dict],
    tier_name: str,
    extractor_fn,
) -> TierMetrics:
    """Run a full benchmark tier against the dataset."""
    metrics = TierMetrics(tier_name=tier_name)

    for item in dataset:
        result = score_sentence(item, extractor_fn, tier_name)
        metrics.sentence_results.append(result)
        metrics.total_sentences += 1
        metrics.total_expected_slots += len(item["expected"])
        metrics.true_positives += result.true_positives
        metrics.false_positives += result.false_positives
        metrics.false_negatives += result.false_negatives
        metrics.total_latency_ms += result.latency_ms

    return metrics


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(
    tiers: List[TierMetrics],
    verbose: bool = False,
    dataset: Optional[List[dict]] = None,
) -> None:
    """Print a formatted benchmark report."""
    print("=" * 78)
    print("  GroundCheck Extraction Benchmark")
    print("=" * 78)

    if dataset:
        styles = {}
        diffs = {}
        for item in dataset:
            s = item.get("style", "unknown")
            d = item.get("difficulty", "unknown")
            styles[s] = styles.get(s, 0) + 1
            diffs[d] = diffs.get(d, 0) + 1
        total = len(dataset)
        total_slots = sum(len(d["expected"]) for d in dataset)
        print(f"\n  Dataset: {total} sentences, {total_slots} expected slots")
        print(f"  Styles:  {', '.join(f'{k}: {v}' for k, v in sorted(styles.items()))}")
        print(f"  Difficulty: {', '.join(f'{k}: {v}' for k, v in sorted(diffs.items()))}")

    print("\n" + "-" * 78)
    print(f"  {'Tier':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Avg ms':>10} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 78)

    for t in tiers:
        print(
            f"  {t.tier_name:<20} "
            f"{t.precision:>9.1%} "
            f"{t.recall:>9.1%} "
            f"{t.f1:>9.1%} "
            f"{t.avg_latency_ms:>9.2f} "
            f"{t.true_positives:>5} "
            f"{t.false_positives:>5} "
            f"{t.false_negatives:>5}"
        )

    print("-" * 78)

    # Improvement summary
    if len(tiers) >= 2:
        regex = tiers[0]
        combined = tiers[-1]
        recall_delta = combined.recall - regex.recall
        f1_delta = combined.f1 - regex.f1
        print(f"\n  Improvement (Combined over Regex-only):")
        print(f"    Recall: {recall_delta:+.1%}")
        print(f"    F1:     {f1_delta:+.1%}")

    # Per-difficulty breakdown
    for t in tiers:
        if not dataset:
            continue
        print(f"\n  {t.tier_name} — by difficulty:")
        for diff in ["easy", "medium", "hard"]:
            results = [
                r for r, d in zip(t.sentence_results, dataset)
                if d.get("difficulty") == diff
            ]
            if not results:
                continue
            tp = sum(r.true_positives for r in results)
            fn = sum(r.false_negatives for r in results)
            fp = sum(r.false_positives for r in results)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r_val = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r_val / (p + r_val) if (p + r_val) > 0 else 0
            print(f"    {diff:<8} P={p:.1%}  R={r_val:.1%}  F1={f1:.1%}  (TP={tp} FP={fp} FN={fn})")

    # Per-style breakdown
    for t in tiers:
        if not dataset:
            continue
        print(f"\n  {t.tier_name} — by style:")
        for style in ["template", "conversational"]:
            results = [
                r for r, d in zip(t.sentence_results, dataset)
                if d.get("style") == style
            ]
            if not results:
                continue
            tp = sum(r.true_positives for r in results)
            fn = sum(r.false_negatives for r in results)
            fp = sum(r.false_positives for r in results)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r_val = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r_val / (p + r_val) if (p + r_val) > 0 else 0
            print(f"    {style:<16} P={p:.1%}  R={r_val:.1%}  F1={f1:.1%}  (TP={tp} FP={fp} FN={fn})")

    if verbose:
        print("\n" + "=" * 78)
        print("  DETAILED PER-SENTENCE RESULTS")
        print("=" * 78)
        for t in tiers:
            print(f"\n  --- {t.tier_name} ---")
            for r in t.sentence_results:
                status = "PASS" if r.false_negatives == 0 and r.false_positives == 0 else "MISS"
                print(f"\n  [{r.id}] {status} ({r.latency_ms:.1f}ms)")
                print(f"    Text: {r.text[:80]}{'...' if len(r.text) > 80 else ''}")
                for ss in r.slot_scores:
                    icon = "+" if ss.correct else "x"
                    got = f"got={ss.extracted_value}" if ss.extracted_value else "MISSING"
                    print(f"    [{icon}] {ss.slot}: expected={ss.expected_value}, {got}")
                if r.false_positives > 0:
                    extra = {k: v for k, v in r.extracted.items()
                             if _normalize_slot(k) not in {_normalize_slot(s) for s in r.expected}}
                    for k, v in extra.items():
                        print(f"    [!] FP: {k}={v}")

    print()


def to_json(tiers: List[TierMetrics], dataset: List[dict]) -> str:
    """Return JSON-serializable benchmark results."""
    results = {}
    for t in tiers:
        results[t.tier_name] = {
            "precision": round(t.precision, 4),
            "recall": round(t.recall, 4),
            "f1": round(t.f1, 4),
            "avg_latency_ms": round(t.avg_latency_ms, 2),
            "true_positives": t.true_positives,
            "false_positives": t.false_positives,
            "false_negatives": t.false_negatives,
            "total_sentences": t.total_sentences,
            "total_expected_slots": t.total_expected_slots,
        }
    meta = {
        "dataset_size": len(dataset),
        "total_expected_slots": sum(len(d["expected"]) for d in dataset),
    }
    return json.dumps({"meta": meta, "results": results}, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GroundCheck Extraction Benchmark")
    parser.add_argument("--style", choices=["template", "conversational"], help="Filter by sentence style")
    parser.add_argument("--category", help="Filter by category (personal, technical, migration, etc.)")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], help="Filter by difficulty")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-sentence detail")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of table")
    args = parser.parse_args()

    dataset = load_dataset(
        style_filter=args.style,
        category_filter=args.category,
        difficulty_filter=args.difficulty,
    )

    if not dataset:
        print("No matching sentences in dataset.")
        return

    # Run all three tiers
    tier_regex = run_benchmark(dataset, "Regex Only", run_regex_only)
    tier_knowledge = run_benchmark(dataset, "Knowledge Only", run_knowledge_only)
    tier_combined = run_benchmark(dataset, "Combined", run_combined)

    tiers = [tier_regex, tier_knowledge, tier_combined]

    if args.json:
        print(to_json(tiers, dataset))
    else:
        print_report(tiers, verbose=args.verbose, dataset=dataset)


if __name__ == "__main__":
    main()
