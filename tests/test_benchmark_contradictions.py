"""Test GroundCheck against contradiction benchmark."""

import json
from pathlib import Path
from groundcheck import GroundCheck, Memory


def run_benchmark():
    """Run GroundCheck against contradiction benchmark."""
    verifier = GroundCheck()
    
    # Path to benchmark data
    data_file = Path("../groundingbench/data/contradictions.jsonl")
    
    if not data_file.exists():
        print(f"Error: Benchmark file not found at {data_file}")
        print("Please ensure groundingbench data is available.")
        return
    
    correct = 0
    total = 0
    results = []
    
    with open(data_file) as f:
        for line in f:
            example = json.loads(line)
            
            # Convert retrieved context to Memory objects
            memories = []
            for ctx in example["retrieved_context"]:
                memory = Memory(
                    id=ctx["id"],
                    text=ctx["text"],
                    trust=ctx.get("trust", 1.0),
                    timestamp=ctx.get("timestamp")
                )
                memories.append(memory)
            
            # Run verification
            result = verifier.verify(example["generated_output"], memories)
            
            # Check against label
            label = example["label"]
            expected_grounded = label.get("grounded", False)
            requires_disclosure = label.get("requires_contradiction_disclosure", False)
            
            # Our assessment
            # If the benchmark says it requires disclosure, we should flag it
            if requires_disclosure:
                # We pass if we correctly identified the need for disclosure
                passed = result.requires_disclosure == requires_disclosure
            else:
                # Standard grounding check
                passed = result.passed == expected_grounded
            
            if passed:
                correct += 1
            else:
                # Track failures for analysis
                results.append({
                    "id": example["id"],
                    "expected": {
                        "grounded": expected_grounded,
                        "requires_disclosure": requires_disclosure
                    },
                    "got": {
                        "passed": result.passed,
                        "requires_disclosure": result.requires_disclosure,
                        "contradicted_claims": result.contradicted_claims
                    }
                })
            
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    
    print("=" * 70)
    print("CONTRADICTION BENCHMARK RESULTS")
    print("=" * 70)
    print(f"\nAccuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"Target: 95%+")
    
    if accuracy >= 0.95:
        print("\n✅ PASSED - Exceeds target accuracy!")
    elif accuracy >= 0.90:
        print("\n⚠️  CLOSE - Near target but below 95%")
    else:
        print(f"\n❌ NEEDS WORK - Below 90% accuracy")
    
    # Show failures if any
    if results:
        print(f"\nFailures ({len(results)}):")
        for r in results[:5]:  # Show first 5 failures
            print(f"  - {r['id']}")
            print(f"    Expected: grounded={r['expected']['grounded']}, "
                  f"requires_disclosure={r['expected']['requires_disclosure']}")
            print(f"    Got: passed={r['got']['passed']}, "
                  f"requires_disclosure={r['got']['requires_disclosure']}")
    
    print("=" * 70)
    
    return accuracy


if __name__ == "__main__":
    run_benchmark()
