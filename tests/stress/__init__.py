"""Stress tests for MLX Primitives.

These tests exercise the library under extreme conditions:
- Very long sequences (16K+)
- OOM scenarios
- Concurrent operations
- Extended stability

All tests are marked with @pytest.mark.stress and @pytest.mark.slow
so they can be selectively run:

    pytest tests/stress/ -v --timeout=300
    pytest -m stress -v
"""
