#!/usr/bin/env python3
"""
Trust-Aware Agent Demo -- Full CRT Pipeline.

Shows: memory storage -> contradiction detection -> trust evolution ->
lifecycle tracking -> disclosure policy -> verification.

Run with:  python examples/trust_aware_agent.py
Requires:  pip install groundcheck
"""

import time
from groundcheck import (
    GroundCheck,
    Memory,
    CRTConfig,
    CRTMath,
    ContradictionLedger,
    ContradictionLifecycle,
    ContradictionLifecycleEntry,
    ContradictionLifecycleState,
    DisclosurePolicy,
    UserTransparencyPrefs,
    TransparencyLevel,
)


def main():
    print("=" * 60)
    print("GroundCheck v2 -- Trust-Aware Agent Demo")
    print("=" * 60)

    # -- 1. Setup ----------------------------------------------
    verifier = GroundCheck()
    config = CRTConfig()
    math = CRTMath(config)

    # In-memory ledger (uses temp DB)
    import tempfile, os
    tmp = tempfile.mkdtemp()
    ledger = ContradictionLedger(db_path=os.path.join(tmp, "demo.db"))

    lifecycle = ContradictionLifecycle()
    prefs = UserTransparencyPrefs(transparency_level=TransparencyLevel.BALANCED)
    disclosure = DisclosurePolicy(user_prefs=prefs, lifecycle=lifecycle)

    # -- 2. Store initial memories with trust ------------------
    memories = [
        Memory(id="m1", text="User works at Microsoft", trust=0.9),
        Memory(id="m2", text="User lives in Seattle", trust=0.85),
        Memory(id="m3", text="User has a dog named Max", trust=0.7),
    ]
    print("\n[Initial Memories]")
    for m in memories:
        print(f"  {m.id}: {m.text} (trust={m.trust})")

    # -- 3. User says something contradictory ------------------
    new_statement = "I just started at Amazon last week"
    print(f"\n[User says]: \"{new_statement}\"")

    # -- 4. GroundCheck detects contradiction ------------------
    result = verifier.verify(new_statement, memories)
    print(f"\n[Verification Result]")
    print(f"  Passed: {result.passed}")
    if result.hallucinations:
        print(f"  Hallucinations: {result.hallucinations}")
    if result.contradiction_details:
        for cd in result.contradiction_details:
            print(f"  Contradiction in slot '{cd.slot}': {cd.values}")

    # -- 5. Record in ledger ----------------------------------─
    entry = ledger.record_contradiction(
        old_memory_id="m1",
        new_memory_id="m_new",
        drift_mean=0.75,
        confidence_delta=0.2,
        old_text="User works at Microsoft",
        new_text="User just started at Amazon",
        query=new_statement,
    )
    print(f"\n[Ledger Entry]")
    print(f"  ID: {entry.ledger_id}")
    print(f"  Type: {entry.contradiction_type}")
    print(f"  Status: {entry.status}")

    # -- 6. Trust evolution ------------------------------------
    old_trust = 0.9
    new_trust = math.evolve_trust_contradicted(old_trust, 0.75)
    print(f"\n[Trust Evolution]")
    print(f"  Microsoft memory: {old_trust} -> {new_trust:.3f} (contradicted)")

    reinforced_trust = math.evolve_trust_reinforced(0.85, 0.1)
    print(f"  Seattle memory: 0.85 -> {reinforced_trust:.3f} (reinforced)")

    # -- 7. Lifecycle tracking --------------------------------─
    lc_entry = ContradictionLifecycleEntry(
        ledger_id=entry.ledger_id,
        affected_slots={"employer"},
        old_value="Microsoft",
        new_value="Amazon",
    )
    print(f"\n[Lifecycle]")
    print(f"  State: {lc_entry.state.value}")

    # Simulate user confirming "Amazon" twice
    lifecycle.record_confirmation(lc_entry)
    lifecycle.record_confirmation(lc_entry)
    print(f"  After 2 confirmations: {lc_entry.state.value}")

    # -- 8. Disclosure decision --------------------------------
    should_tell = disclosure.should_disclose(lc_entry)
    print(f"\n[Disclosure Policy]")
    print(f"  Should disclose: {should_tell}")

    # -- 9. Verify output using updated trust ------------------
    updated_memories = [
        Memory(id="m1", text="User works at Microsoft", trust=new_trust),
        Memory(id="m_new", text="User started at Amazon", trust=0.8),
        Memory(id="m2", text="User lives in Seattle", trust=0.85),
    ]
    output = "You work at Microsoft and live in Seattle."
    final = verifier.verify(output, updated_memories)
    print(f"\n[Final Verification]")
    print(f"  Text: \"{output}\"")
    print(f"  Passed: {final.passed}")
    print(f"  Confidence: {final.confidence:.2f}")
    if final.requires_disclosure:
        print(f"  WARNING: Requires disclosure -- conflicting employer facts")

    # -- 10. Stats --------------------------------------------─
    stats = ledger.get_contradiction_stats(days=1)
    print(f"\n[Ledger Stats]")
    print(f"  Total: {stats['total_contradictions']}")
    print(f"  Open: {stats['open']}")

    print("\n" + "=" * 60)
    print("Demo complete. All CRT modules working.")
    print("=" * 60)

    # Cleanup
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
