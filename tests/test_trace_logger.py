"""Tests for groundcheck.trace_logger."""

import os
import tempfile
import logging
import pytest
from groundcheck.trace_logger import (
    ContradictionTraceLogger,
    get_trace_logger,
    configure_trace_logging,
    _global_trace_logger,
)


class TestTraceLogger:
    def test_creation_with_defaults(self):
        logger = ContradictionTraceLogger(log_file=None, console_output=False)
        assert logger.logger is not None

    def test_log_file_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "subdir", "test_trace.log")
            tl = ContradictionTraceLogger(log_file=log_file, console_output=False)
            tl.log_contradiction_detected(
                ledger_id="c1",
                old_memory_id="m1",
                new_memory_id="m2",
                old_text="works at Google",
                new_text="works at Microsoft",
                drift_mean=0.75,
                contradiction_type="CONFLICT",
            )
            # Flush and close handlers so Windows releases file locks
            for h in tl.logger.handlers:
                h.flush()
                h.close()
            assert os.path.exists(log_file)
            with open(log_file) as f:
                content = f.read()
            assert "CONTRADICTION DETECTED" in content
            assert "c1" in content

    def test_structured_events(self):
        logger = ContradictionTraceLogger(log_file=None, console_output=False)
        # These should not raise
        logger.log_resolution_attempt("I work at Microsoft", [], 2)
        logger.log_resolution_matched("c1", "CONFLICT", "employer", "Google", "Microsoft", "Microsoft", "user_chose_new")
        logger.log_ledger_update("c1", "open", "resolved", "user_chose_new", "m2")
        logger.log_resolution_complete("c1", True, "user chose new value")
        logger.log_resolution_summary(5, 3, 2, 0.5)
        logger.log_pattern_statistics({"pattern1": 5, "pattern2": 3}, 8)


class TestSingleton:
    def test_get_trace_logger_creates_instance(self):
        import groundcheck.trace_logger as mod
        mod._global_trace_logger = None  # Reset singleton
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "subdir", "trace.log")
            tl = get_trace_logger(log_file=log_file, console_output=False)
            assert tl is not None
            # Calling again returns same instance
            tl2 = get_trace_logger()
            assert tl2 is tl
            # Close handlers to release file locks on Windows
            for h in tl.logger.handlers:
                h.close()
        mod._global_trace_logger = None  # Clean up

    def test_configure_disables_logging(self):
        import groundcheck.trace_logger as mod
        mod._global_trace_logger = None
        configure_trace_logging(enabled=False)
        assert mod._global_trace_logger is None
        mod._global_trace_logger = None  # Clean up
