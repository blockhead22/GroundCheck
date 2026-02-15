"""Latency benchmark for GroundCheck verification.

Measures real-world latency across multiple scenarios:
- Simple verification (1-3 memories)
- Complex verification (10+ memories with contradictions)
- Fact extraction only
- Knowledge extraction only

Usage:
    python -m benchmarks.latency
    python benchmarks/latency.py
"""

import json
import statistics
import time
from typing import List, Dict, Any

from groundcheck import GroundCheck, Memory
from groundcheck.fact_extractor import extract_fact_slots
from groundcheck.knowledge_extractor import extract_knowledge_facts


def _timed(fn, iterations: int = 1000) -> Dict[str, float]:
    """Run fn() `iterations` times and return latency stats in ms."""
    times: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

    times.sort()
    return {
        "mean_ms": round(statistics.mean(times), 3),
        "median_ms": round(statistics.median(times), 3),
        "p95_ms": round(times[int(len(times) * 0.95)], 3),
        "p99_ms": round(times[int(len(times) * 0.99)], 3),
        "min_ms": round(times[0], 3),
        "max_ms": round(times[-1], 3),
        "iterations": iterations,
    }


def bench_simple_verify(verifier: GroundCheck) -> Dict[str, float]:
    """Verify a short sentence against 2 memories."""
    memories = [
        Memory(id="m1", text="User works at Microsoft", trust=0.9),
        Memory(id="m2", text="User lives in Seattle", trust=0.8),
    ]
    text = "You work at Amazon and live in Seattle"
    return _timed(lambda: verifier.verify(text, memories))


def bench_complex_verify(verifier: GroundCheck) -> Dict[str, float]:
    """Verify against 10 memories with contradictions."""
    memories = [
        Memory(id="m1", text="User works at Microsoft", trust=0.9),
        Memory(id="m2", text="User lives in Seattle", trust=0.85),
        Memory(id="m3", text="User's name is Alice", trust=0.95),
        Memory(id="m4", text="User uses Python and TypeScript", trust=0.8),
        Memory(id="m5", text="User's database is PostgreSQL", trust=0.7),
        Memory(id="m6", text="User works at Google", trust=0.3),  # contradiction
        Memory(id="m7", text="User's age is 32", trust=0.6),
        Memory(id="m8", text="User graduated from MIT", trust=0.75),
        Memory(id="m9", text="User's hobby is hiking", trust=0.5),
        Memory(id="m10", text="User's framework is FastAPI", trust=0.8),
    ]
    text = (
        "Alice, you work at Google and live in Portland. "
        "You're 32 and use React with MongoDB."
    )
    return _timed(lambda: verifier.verify(text, memories))


def bench_extract_regex() -> Dict[str, float]:
    """Regex-based fact extraction on a multi-claim sentence."""
    text = (
        "My name is Bob, I'm 28, I work at Tesla, "
        "I live in Austin, and I use Rust and Go."
    )
    return _timed(lambda: extract_fact_slots(text))


def bench_extract_knowledge() -> Dict[str, float]:
    """Knowledge-based extraction on conversational text."""
    text = (
        "Yeah so we ended up going with Postgres after the whole MySQL disaster. "
        "We're also considering switching to Kubernetes for orchestration."
    )
    return _timed(lambda: extract_knowledge_facts(text))


def bench_many_memories(verifier: GroundCheck, n: int = 50) -> Dict[str, float]:
    """Verify against a large memory set."""
    memories = [
        Memory(id=f"m{i}", text=f"User fact number {i} is value_{i}", trust=0.5 + 0.01 * i)
        for i in range(n)
    ]
    text = "User fact number 5 is value_99 and fact number 10 is value_10"
    return _timed(lambda: verifier.verify(text, memories), iterations=500)


def main():
    print("GroundCheck Latency Benchmark")
    print("=" * 50)
    print()

    verifier = GroundCheck(neural=False)

    benchmarks = {
        "simple_verify (2 memories)": lambda: bench_simple_verify(verifier),
        "complex_verify (10 memories, contradictions)": lambda: bench_complex_verify(verifier),
        "regex_extraction": lambda: bench_extract_regex,
        "knowledge_extraction": lambda: bench_extract_knowledge,
        "large_memory_set (50 memories)": lambda: bench_many_memories(verifier, 50),
    }

    # Fix: call the bench functions properly
    results: Dict[str, Any] = {}

    print("Running: simple_verify (2 memories) × 1000 ...")
    r = bench_simple_verify(verifier)
    results["simple_verify"] = r
    print(f"  mean={r['mean_ms']:.3f}ms  p95={r['p95_ms']:.3f}ms  p99={r['p99_ms']:.3f}ms")

    print("Running: complex_verify (10 memories) × 1000 ...")
    r = bench_complex_verify(verifier)
    results["complex_verify"] = r
    print(f"  mean={r['mean_ms']:.3f}ms  p95={r['p95_ms']:.3f}ms  p99={r['p99_ms']:.3f}ms")

    print("Running: regex_extraction × 1000 ...")
    r = bench_extract_regex()
    results["regex_extraction"] = r
    print(f"  mean={r['mean_ms']:.3f}ms  p95={r['p95_ms']:.3f}ms  p99={r['p99_ms']:.3f}ms")

    print("Running: knowledge_extraction × 1000 ...")
    r = bench_extract_knowledge()
    results["knowledge_extraction"] = r
    print(f"  mean={r['mean_ms']:.3f}ms  p95={r['p95_ms']:.3f}ms  p99={r['p99_ms']:.3f}ms")

    print("Running: large_memory_set (50 memories) × 500 ...")
    r = bench_many_memories(verifier, 50)
    results["large_memory_set_50"] = r
    print(f"  mean={r['mean_ms']:.3f}ms  p95={r['p95_ms']:.3f}ms  p99={r['p99_ms']:.3f}ms")

    print()
    print("=" * 50)
    print("Summary (all regex-only, neural=False):")
    print()
    for name, r in results.items():
        print(f"  {name:30s}  mean={r['mean_ms']:7.3f}ms  p95={r['p95_ms']:7.3f}ms")

    # Save results
    output_path = "benchmarks/latency_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
