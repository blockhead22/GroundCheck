"""
Trust Decay & Reinforcement System.

Implements temporal trust dynamics using CRT's mathematical framework:

1. **Decay**: Exponential recency curve instead of flat subtract.
   Memories far from the time constant decay faster; recently active ones barely decay.
2. **Reinforcement**: Drift-aware trust boost — semantically close references
   give bigger boosts than tangential ones.
3. **Corrections**: Aligned trust evolution for user corrections.

Constraints:
- Never decay below TRUST_FLOOR (0.20)
- Never boost above TRUST_CEILING (0.95)
- Grace period: new memories are immune for GRACE_PERIOD_DAYS
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# Tuning constants
DECAY_RATE = 0.02
REINFORCE_BOOST = 0.03
CORRECTION_BOOST = 0.05
TRUST_FLOOR = 0.20
TRUST_CEILING = 0.95
GRACE_PERIOD_DAYS = 7
MIN_PASS_INTERVAL_SECS = 3600

# Module-level state
_last_decay_ts: float = 0.0
_crt_math = None


def _get_crt_math():
    """Lazy-load CRTMath singleton."""
    global _crt_math
    if _crt_math is not None:
        return _crt_math
    try:
        from .trust_math import CRTMath, CRTConfig
        _crt_math = CRTMath(CRTConfig())
        logger.info("[TRUST_DECAY] CRT math loaded")
        return _crt_math
    except Exception as e:
        logger.debug("[TRUST_DECAY] CRT math unavailable (%s)", e)
        return None


def _compute_exponential_decay(age_seconds: float, current_trust: float) -> float:
    """CRT-style exponential decay.

    Uses recency curve rho = e^{-dt/lambda} with 30x time constant scaling.
    """
    crt = _get_crt_math()
    if crt is None:
        return max(TRUST_FLOOR, current_trust - DECAY_RATE)

    lambda_decay = crt.config.lambda_time * 30.0
    rho = math.exp(-age_seconds / lambda_decay)
    new_trust = TRUST_FLOOR + (current_trust - TRUST_FLOOR) * rho
    decay_this_pass = (current_trust - new_trust) * 0.1
    return max(TRUST_FLOOR, current_trust - decay_this_pass)


def _compute_drift_aware_boost(
    current_trust: float,
    memory_text: str,
    context_text: str,
    is_correction: bool = False,
    encoder: Optional[Callable] = None,
) -> float:
    """CRT drift-aware trust boost.

    Low drift -> bigger boost, high drift -> smaller boost.
    """
    crt = _get_crt_math()
    if crt is None:
        return min(TRUST_CEILING, current_trust + (CORRECTION_BOOST if is_correction else REINFORCE_BOOST))

    drift = 0.1  # Default low drift
    if memory_text and context_text and encoder is not None:
        try:
            vec_mem = encoder(memory_text)
            vec_ctx = encoder(context_text)
            if vec_mem is not None and vec_ctx is not None:
                drift = crt.drift_meaning(vec_ctx, vec_mem)
        except Exception:
            pass

    if is_correction:
        new_trust = crt.evolve_trust_aligned(current_trust, drift)
    else:
        new_trust = crt.evolve_trust_reinforced(current_trust, drift)

    return min(TRUST_CEILING, max(TRUST_FLOOR, new_trust))


def _find_groundcheck_db(db_path: Optional[str] = None) -> Optional[Path]:
    """Locate the memory database."""
    if db_path:
        p = Path(db_path)
        if p.is_file():
            return p

    env = os.environ.get("GROUNDCHECK_DB", "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return p

    candidates = [
        Path.home() / ".groundcheck" / "memory.db",
        Path(".groundcheck") / "memory.db",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _is_crt_schema(conn: sqlite3.Connection) -> bool:
    """Return True if DB uses CRT memory schema (memory_id PK)."""
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()]
        return "memory_id" in cols
    except Exception:
        return False


def run_trust_decay_pass(
    db_path: Optional[str] = None,
    encoder: Optional[Callable] = None,
) -> dict:
    """Run one trust decay + reinforcement pass.

    Args:
        db_path: Path to SQLite memory database. If None, auto-discovers.
        encoder: Optional callable ``(text) -> vector`` for drift computation.

    Returns:
        Summary dict with counts of decayed/reinforced/skipped.
    """
    global _last_decay_ts

    now = time.time()
    if (now - _last_decay_ts) < MIN_PASS_INTERVAL_SECS:
        return {"skipped": True, "reason": "too_soon"}

    found = _find_groundcheck_db(db_path)
    if not found:
        return {"skipped": True, "reason": "db_not_found"}

    grace_cutoff = now - (GRACE_PERIOD_DAYS * 86400)
    crt_available = _get_crt_math() is not None

    try:
        conn = sqlite3.connect(str(found))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")

        crt_schema = _is_crt_schema(conn)
        id_col = "memory_id" if crt_schema else "id"
        alive_clause = "AND (deprecated IS NULL OR deprecated = 0)" if crt_schema else ""

        # 1. Decay
        stale_rows = conn.execute(
            f"""SELECT {id_col} AS mem_id, trust, timestamp, text FROM memories
               WHERE timestamp < ? AND trust > ? {alive_clause}""",
            (grace_cutoff, TRUST_FLOOR),
        ).fetchall()

        decayed = 0
        for row in stale_rows:
            age_seconds = now - float(row["timestamp"])
            new_trust = _compute_exponential_decay(age_seconds, row["trust"])
            if new_trust < row["trust"]:
                conn.execute(
                    f"UPDATE memories SET trust = ? WHERE {id_col} = ?",
                    (round(new_trust, 4), row["mem_id"]),
                )
                decayed += 1

        # 2. Reinforce from copilot_events
        reinforced = 0
        has_events = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='copilot_events'"
        ).fetchone()[0]

        if has_events:
            recent_events = conn.execute(
                """SELECT memory_id, event_type, new_text FROM copilot_events
                   WHERE timestamp > ?""",
                (int(_last_decay_ts) if _last_decay_ts > 0 else int(now - 86400),),
            ).fetchall()

            for evt in recent_events:
                mem = conn.execute(
                    f"SELECT trust, text FROM memories WHERE {id_col} = ?",
                    (evt["memory_id"],),
                ).fetchone()
                if not mem:
                    continue

                is_correction = evt["event_type"] == "correction"
                context_text = ""
                try:
                    context_text = evt["new_text"] or ""
                except (IndexError, KeyError):
                    pass

                if crt_available and mem["text"]:
                    new_trust = _compute_drift_aware_boost(
                        mem["trust"], mem["text"],
                        context_text or mem["text"],
                        is_correction=is_correction,
                        encoder=encoder,
                    )
                else:
                    boost = CORRECTION_BOOST if is_correction else REINFORCE_BOOST
                    new_trust = min(TRUST_CEILING, mem["trust"] + boost)

                if new_trust > mem["trust"]:
                    conn.execute(
                        f"UPDATE memories SET trust = ? WHERE {id_col} = ?",
                        (round(new_trust, 4), evt["memory_id"]),
                    )
                    reinforced += 1

        conn.commit()
        conn.close()

        _last_decay_ts = now
        summary = {
            "skipped": False,
            "decayed": decayed,
            "reinforced": reinforced,
            "total_stale_checked": len(stale_rows),
            "crt_math_active": crt_available,
            "timestamp": now,
        }
        logger.info("[TRUST_DECAY] Pass complete: %s", summary)
        return summary

    except Exception as e:
        logger.warning("[TRUST_DECAY] Error during decay pass: %s", e)
        return {"skipped": True, "reason": str(e)}


def reinforce_memory(
    memory_id: str,
    boost: float = REINFORCE_BOOST,
    context_text: str = "",
    is_correction: bool = False,
    db_path: Optional[str] = None,
    encoder: Optional[Callable] = None,
) -> bool:
    """Give a specific memory a trust boost.

    If context_text is provided and CRT math is available, computes semantic
    drift to scale the boost. Falls back to flat boost otherwise.

    Args:
        memory_id: The memory to reinforce.
        boost: Flat fallback boost amount.
        context_text: Optional text for drift computation.
        is_correction: Whether this is a user correction.
        db_path: Path to SQLite memory database.
        encoder: Optional callable ``(text) -> vector`` for drift computation.
    """
    found = _find_groundcheck_db(db_path)
    if not found:
        return False

    try:
        conn = sqlite3.connect(str(found))
        conn.row_factory = sqlite3.Row
        id_col = "memory_id" if _is_crt_schema(conn) else "id"
        row = conn.execute(
            f"SELECT trust, text FROM memories WHERE {id_col} = ?",
            (memory_id,),
        ).fetchone()
        if not row:
            conn.close()
            return False

        crt = _get_crt_math()
        if crt is not None and row["text"] and context_text:
            new_trust = _compute_drift_aware_boost(
                row["trust"], row["text"], context_text,
                is_correction=is_correction, encoder=encoder,
            )
        else:
            new_trust = min(TRUST_CEILING, row["trust"] + boost)

        if new_trust > row["trust"]:
            conn.execute(
                f"UPDATE memories SET trust = ? WHERE {id_col} = ?",
                (round(new_trust, 4), memory_id),
            )
            conn.commit()

        conn.close()
        return True
    except Exception as e:
        logger.warning("[TRUST_DECAY] reinforce_memory failed: %s", e)
        return False
