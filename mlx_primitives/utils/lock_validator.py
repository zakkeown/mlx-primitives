"""Lock order validator for debugging deadlock-prone lock hierarchies.

This module provides runtime validation of lock acquisition order to detect
potential deadlocks early. The validation is only active in debug mode
(__debug__ == True) to avoid production overhead.

Lock Ordering Protocol for Cache Components:
    1. PageTable._lock (RLock) - highest priority
    2. BlockAllocator._lock (RLock)
    3. EvictionPolicy._lock (Lock) - lowest priority

Violations of this order will raise RuntimeError in debug mode.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Generator


class LockOrderValidator:
    """Validates lock acquisition order at runtime (debug mode only).

    This class tracks which locks are held by each thread and validates
    that locks are always acquired in the correct order to prevent deadlocks.

    Usage:
        # In classes that use locks:
        def some_method(self):
            LockOrderValidator.validate_acquire("PageTable")
            with self._lock:
                try:
                    # ... do work ...
                finally:
                    pass
            LockOrderValidator.validate_release("PageTable")

        # Or use the context manager:
        with ordered_lock("PageTable", self._lock):
            # ... do work ...
    """

    _thread_local = threading.local()

    # Lock hierarchy: higher number = must acquire first
    LOCK_LEVELS = {
        "PageTable": 3,
        "BlockAllocator": 2,
        "EvictionPolicy": 1,
    }

    @classmethod
    def validate_acquire(cls, lock_name: str) -> None:
        """Validate that acquiring this lock respects the lock hierarchy.

        Args:
            lock_name: Name of the lock being acquired.

        Raises:
            RuntimeError: If acquiring this lock would violate the lock order.
        """
        if not __debug__:
            return

        if not hasattr(cls._thread_local, "held"):
            cls._thread_local.held = []

        my_level = cls.LOCK_LEVELS.get(lock_name, 0)

        for held_name, held_level in cls._thread_local.held:
            if held_level < my_level:
                raise RuntimeError(
                    f"Lock order violation: attempting to acquire '{lock_name}' "
                    f"(level {my_level}) while holding '{held_name}' (level {held_level}). "
                    f"This can cause deadlocks. See page_table.py for lock ordering protocol."
                )

        cls._thread_local.held.append((lock_name, my_level))

    @classmethod
    def validate_release(cls, lock_name: str) -> None:
        """Record that a lock has been released.

        Args:
            lock_name: Name of the lock being released.
        """
        if not __debug__:
            return

        if not hasattr(cls._thread_local, "held"):
            cls._thread_local.held = []
            return

        cls._thread_local.held = [
            (n, l) for n, l in cls._thread_local.held if n != lock_name
        ]

    @classmethod
    def get_held_locks(cls) -> list[tuple[str, int]]:
        """Get list of currently held locks for this thread (for debugging)."""
        if not hasattr(cls._thread_local, "held"):
            return []
        return list(cls._thread_local.held)

    @classmethod
    def clear_thread_state(cls) -> None:
        """Clear all lock tracking state for the current thread.

        This should be called when a thread is done with lock-protected
        operations, especially in long-running threads, to prevent potential
        memory leaks from accumulated tracking state.

        Example:
            # At the end of a worker thread's lifecycle:
            LockOrderValidator.clear_thread_state()
        """
        if hasattr(cls._thread_local, "held"):
            cls._thread_local.held.clear()


@contextmanager
def ordered_lock(
    name: str, lock: Any, validate: bool = __debug__
) -> Generator[None, None, None]:
    """Context manager that validates lock order in debug mode.

    Args:
        name: Name of the lock for hierarchy validation.
        lock: The actual lock object to acquire.
        validate: Whether to validate lock order (default: __debug__).

    Yields:
        None

    Raises:
        RuntimeError: If acquiring this lock would violate the lock order.

    Example:
        with ordered_lock("PageTable", self._lock):
            # ... critical section ...
    """
    if validate:
        LockOrderValidator.validate_acquire(name)
    lock.acquire()
    try:
        yield
    finally:
        lock.release()
        if validate:
            LockOrderValidator.validate_release(name)
