"""
Contradiction Resolution Trace Logger.

Provides detailed trace logging for contradiction resolution events.
Supports structured event logging, configurable output, and singleton access.

Default log directory: ``~/.groundcheck/logs/`` (override via ``GROUNDCHECK_LOG_DIR``).
"""

import logging
import os
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path


def _default_log_dir() -> str:
    """Return the default log directory, respecting GROUNDCHECK_LOG_DIR env var."""
    env_dir = os.environ.get("GROUNDCHECK_LOG_DIR")
    if env_dir:
        return env_dir
    return str(Path.home() / ".groundcheck" / "logs")


class ContradictionTraceLogger:
    """Dedicated logger for contradiction resolution events."""

    def __init__(
        self,
        log_file: Optional[str] = None,
        console_output: bool = True,
        log_level: int = logging.DEBUG,
    ):
        self.logger = logging.getLogger(f"crt.contradiction_trace.{id(self)}")
        self.logger.setLevel(log_level)

        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter("%(levelname)-8s | %(message)s")
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

    def log_contradiction_detected(
        self,
        ledger_id: str,
        old_memory_id: str,
        new_memory_id: str,
        old_text: str,
        new_text: str,
        drift_mean: float,
        contradiction_type: str,
        affected_slots: Optional[List[str]] = None,
    ):
        slots_str = ", ".join(affected_slots) if affected_slots else "none"
        self.logger.info("=== CONTRADICTION DETECTED ===")
        self.logger.info("Ledger ID: %s", ledger_id)
        self.logger.info("Type: %s", contradiction_type)
        self.logger.info("Drift: %.4f", drift_mean)
        self.logger.info("Affected Slots: %s", slots_str)
        self.logger.debug("Old Memory [%s]: %s", old_memory_id, old_text[:100])
        self.logger.debug("New Memory [%s]: %s", new_memory_id, new_text[:100])

    def log_resolution_attempt(
        self,
        user_text: str,
        matched_patterns: List[Dict[str, Any]],
        open_contradictions_count: int,
    ):
        self.logger.info("=== RESOLUTION ATTEMPT ===")
        self.logger.info("User Input: %s", user_text[:150])
        self.logger.info("Open Contradictions: %d", open_contradictions_count)
        if matched_patterns:
            self.logger.info("Matched Patterns (%d):", len(matched_patterns))
            for i, pi in enumerate(matched_patterns[:3], 1):
                self.logger.info("  %d. Pattern: %s", i, pi.get("pattern", "unknown")[:50])
                self.logger.info("     Match: '%s'", pi.get("match", ""))

    def log_resolution_matched(
        self,
        ledger_id: str,
        contradiction_type: str,
        slot_name: Optional[str],
        old_value: Any,
        new_value: Any,
        chosen_value: Any,
        resolution_method: str,
    ):
        self.logger.info("=== RESOLUTION MATCHED ===")
        self.logger.info("Ledger ID: %s", ledger_id)
        self.logger.info("Type: %s", contradiction_type)
        if slot_name:
            self.logger.info("Slot: %s", slot_name)
            self.logger.info("  Old Value: %s", old_value)
            self.logger.info("  New Value: %s", new_value)
            self.logger.info("  Chosen: %s", chosen_value)
        self.logger.info("Resolution Method: %s", resolution_method)

    def log_ledger_update(
        self,
        ledger_id: str,
        before_status: str,
        after_status: str,
        resolution_method: str,
        chosen_memory_id: Optional[str] = None,
    ):
        self.logger.info("=== LEDGER UPDATE ===")
        self.logger.info("Ledger ID: %s", ledger_id)
        self.logger.info("Status: %s -> %s", before_status, after_status)
        self.logger.info("Method: %s", resolution_method)
        if chosen_memory_id:
            self.logger.info("Chosen Memory: %s", chosen_memory_id)

    def log_resolution_complete(
        self,
        ledger_id: str,
        success: bool,
        details: Optional[str] = None,
    ):
        status = "SUCCESS" if success else "FAILED"
        self.logger.info("=== RESOLUTION %s ===", status)
        self.logger.info("Ledger ID: %s", ledger_id)
        if details:
            self.logger.info("Details: %s", details)

    def log_resolution_summary(
        self,
        total_open_before: int,
        total_open_after: int,
        resolved_count: int,
        elapsed_time: float,
    ):
        self.logger.info("=== RESOLUTION SUMMARY ===")
        self.logger.info("Open Before: %d", total_open_before)
        self.logger.info("Open After: %d", total_open_after)
        self.logger.info("Resolved: %d", resolved_count)
        self.logger.info("Time: %.3fs", elapsed_time)

    def log_pattern_statistics(
        self,
        pattern_usage: Dict[str, int],
        total_resolutions: int,
    ):
        self.logger.info("=== PATTERN STATISTICS ===")
        self.logger.info("Total Resolutions: %d", total_resolutions)
        if pattern_usage:
            self.logger.info("Most Used Patterns:")
            sorted_patterns = sorted(
                pattern_usage.items(), key=lambda x: x[1], reverse=True,
            )
            for pattern, count in sorted_patterns[:10]:
                pct = (count / total_resolutions * 100) if total_resolutions > 0 else 0
                self.logger.info("  %s: %d (%.1f%%)", pattern[:50], count, pct)


# Global singleton
_global_trace_logger: Optional[ContradictionTraceLogger] = None


def get_trace_logger(
    log_file: Optional[str] = None,
    console_output: bool = False,
    log_level: int = logging.INFO,
) -> ContradictionTraceLogger:
    """Get or create the global trace logger.

    If ``log_file`` is not provided, defaults to
    ``~/.groundcheck/logs/contradiction_trace.log``
    (or ``GROUNDCHECK_LOG_DIR`` env var).
    """
    global _global_trace_logger

    if _global_trace_logger is None:
        if log_file is None:
            log_file = os.path.join(_default_log_dir(), "contradiction_trace.log")
        _global_trace_logger = ContradictionTraceLogger(
            log_file=log_file,
            console_output=console_output,
            log_level=log_level,
        )

    return _global_trace_logger


def configure_trace_logging(
    enabled: bool = True,
    log_file: Optional[str] = None,
    console_output: bool = False,
    log_level: int = logging.INFO,
):
    """Configure global trace logging settings."""
    global _global_trace_logger

    if enabled:
        if log_file is None:
            log_file = os.path.join(_default_log_dir(), "contradiction_trace.log")
        _global_trace_logger = ContradictionTraceLogger(
            log_file=log_file,
            console_output=console_output,
            log_level=log_level,
        )
    else:
        _global_trace_logger = None
