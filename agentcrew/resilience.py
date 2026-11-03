"""Retry/backoff decorator used to wrap flaky operations (e.g. real LLM API calls).

``retry_with_backoff`` implements exponential backoff with optional jitter,
retrying only on the configured exception types and re-raising the last
exception once ``max_attempts`` is exhausted. ``sleep_fn`` is injectable so
tests can assert retry behavior (attempt counts, which exceptions are
retried) without actually sleeping.
"""

from __future__ import annotations

import functools
import random
import time
from typing import Callable, Tuple, Type, TypeVar

F = TypeVar("F", bound=Callable)


def retry_with_backoff(
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    max_attempts: int = 3,
    base_delay_seconds: float = 0.5,
    max_delay_seconds: float = 8.0,
    jitter: bool = True,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Callable[[F], F]:
    """Return a decorator retrying the wrapped callable on ``exceptions``.

    The delay before attempt *n* (1-indexed, so the first retry is before
    attempt 2) is ``min(base_delay_seconds * 2 ** (n - 2), max_delay_seconds)``,
    optionally multiplied by a uniform random jitter factor in ``[0.5, 1.5)``
    when ``jitter`` is True. Exceptions not in ``exceptions`` propagate
    immediately without any retry.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # noqa: BLE001 - intentional broad catch of configured types
                    if attempt >= max_attempts:
                        raise
                    delay = min(base_delay_seconds * (2 ** (attempt - 1)), max_delay_seconds)
                    if jitter:
                        delay *= random.uniform(0.5, 1.5)
                    sleep_fn(delay)
                    attempt += 1

        return wrapper  # type: ignore[return-value]

    return decorator
