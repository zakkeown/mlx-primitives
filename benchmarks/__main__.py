"""Entry point for running benchmarks as a module.

Usage:
    python -m benchmarks --all
    python -m benchmarks --attention --quick
    python -m benchmarks --layers --charts
"""

from benchmarks.suite import main

if __name__ == "__main__":
    main()
