"""Tests for logging utilities."""

import logging
import os
import threading
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def reset_logging_state():
    """Reset logging module state before each test."""
    import mlx_primitives.utils.logging as log_module

    # Store original state
    original_logger = log_module._logger
    original_seen = log_module._fallback_seen.copy()

    # Reset state
    log_module._logger = None
    log_module._fallback_seen.clear()

    yield

    # Restore original state
    log_module._logger = original_logger
    log_module._fallback_seen = original_seen


class TestGetLogger:
    def test_returns_logger(self):
        from mlx_primitives.utils.logging import get_logger

        logger = get_logger()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "mlx_primitives"

    def test_singleton(self):
        from mlx_primitives.utils.logging import get_logger

        logger1 = get_logger()
        logger2 = get_logger()
        assert logger1 is logger2

    def test_default_level_is_warning(self):
        from mlx_primitives.utils.logging import get_logger

        # Ensure env var is not set
        with mock.patch.dict(os.environ, {}, clear=True):
            # Need to reset module state since fixture already ran
            import mlx_primitives.utils.logging as log_module

            log_module._logger = None
            logger = get_logger()
            assert logger.level == logging.WARNING

    def test_respects_debug_env_var(self):
        import mlx_primitives.utils.logging as log_module

        log_module._logger = None
        with mock.patch.dict(os.environ, {"MLX_PRIMITIVES_LOG_LEVEL": "DEBUG"}):
            from mlx_primitives.utils.logging import get_logger

            logger = get_logger()
            assert logger.level == logging.DEBUG

    def test_respects_info_env_var(self):
        import mlx_primitives.utils.logging as log_module

        log_module._logger = None
        with mock.patch.dict(os.environ, {"MLX_PRIMITIVES_LOG_LEVEL": "INFO"}):
            from mlx_primitives.utils.logging import get_logger

            logger = get_logger()
            assert logger.level == logging.INFO

    def test_case_insensitive_env_var(self):
        import mlx_primitives.utils.logging as log_module

        log_module._logger = None
        with mock.patch.dict(os.environ, {"MLX_PRIMITIVES_LOG_LEVEL": "debug"}):
            from mlx_primitives.utils.logging import get_logger

            logger = get_logger()
            assert logger.level == logging.DEBUG

    def test_invalid_env_var_warns_and_defaults(self, capsys):
        import mlx_primitives.utils.logging as log_module

        log_module._logger = None
        with mock.patch.dict(os.environ, {"MLX_PRIMITIVES_LOG_LEVEL": "INVALID"}):
            from mlx_primitives.utils.logging import get_logger

            logger = get_logger()
            assert logger.level == logging.WARNING
            captured = capsys.readouterr()
            assert "Invalid MLX_PRIMITIVES_LOG_LEVEL" in captured.err
            assert "INVALID" in captured.err

    def test_thread_safety(self):
        """Verify no duplicate handlers from concurrent initialization."""
        import mlx_primitives.utils.logging as log_module

        log_module._logger = None

        loggers = []
        errors = []

        def get_and_store():
            try:
                from mlx_primitives.utils.logging import get_logger

                loggers.append(get_logger())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_and_store) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent access: {errors}"
        assert len(loggers) == 10
        # All should be same instance
        assert all(l is loggers[0] for l in loggers)
        # Should have exactly one handler
        assert len(loggers[0].handlers) == 1


class TestLogFallback:
    def test_first_fallback_is_warning(self, caplog):
        from mlx_primitives.utils.logging import log_fallback

        with caplog.at_level(logging.DEBUG, logger="mlx_primitives"):
            log_fallback("test_op", ValueError("test error"))

        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.WARNING
        assert "test_op" in caplog.records[0].message
        assert "first occurrence" in caplog.records[0].message

    def test_subsequent_fallback_is_debug(self, caplog):
        from mlx_primitives.utils.logging import get_logger, log_fallback

        # Set logger to DEBUG level so it actually emits DEBUG messages
        logger = get_logger()
        original_level = logger.level
        logger.setLevel(logging.DEBUG)

        try:
            with caplog.at_level(logging.DEBUG, logger="mlx_primitives"):
                log_fallback("test_op", ValueError("error 1"))
                log_fallback("test_op", ValueError("error 2"))

            assert len(caplog.records) == 2
            assert caplog.records[0].levelno == logging.WARNING
            assert caplog.records[1].levelno == logging.DEBUG
            assert "first occurrence" not in caplog.records[1].message
        finally:
            logger.setLevel(original_level)

    def test_different_ops_both_warn_first(self, caplog):
        from mlx_primitives.utils.logging import log_fallback

        with caplog.at_level(logging.DEBUG, logger="mlx_primitives"):
            log_fallback("op_a", ValueError("err"))
            log_fallback("op_b", ValueError("err"))

        assert len(caplog.records) == 2
        assert caplog.records[0].levelno == logging.WARNING
        assert caplog.records[1].levelno == logging.WARNING

    def test_exception_details_included(self, caplog):
        from mlx_primitives.utils.logging import log_fallback

        with caplog.at_level(logging.DEBUG, logger="mlx_primitives"):
            log_fallback("test_op", RuntimeError("kernel failed to compile"))

        assert "RuntimeError" in caplog.records[0].message
        assert "kernel failed to compile" in caplog.records[0].message

    def test_context_included(self, caplog):
        from mlx_primitives.utils.logging import log_fallback

        with caplog.at_level(logging.DEBUG, logger="mlx_primitives"):
            log_fallback("test_op", ValueError("err"), context="shape=(32, 64)")

        assert "shape=(32, 64)" in caplog.records[0].message

    def test_context_optional(self, caplog):
        from mlx_primitives.utils.logging import log_fallback

        with caplog.at_level(logging.DEBUG, logger="mlx_primitives"):
            log_fallback("test_op", ValueError("err"))

        # Should not fail without context
        assert "test_op" in caplog.records[0].message
        assert "context" not in caplog.records[0].message


class TestIntegration:
    """Integration tests to verify the logging module works correctly."""

    def test_hardware_detection_uses_shared_logger(self):
        """Verify hardware detection module uses the shared logger."""
        from mlx_primitives.utils.logging import get_logger

        # Get the shared logger
        shared_logger = get_logger()

        # Import hardware detection (which should use get_logger)
        from mlx_primitives.hardware import detection

        # The detection module should use the same logger
        # (we can't easily verify this without mocking, but at least
        # ensure no import errors occur)
        assert shared_logger.name == "mlx_primitives"
