"""Tests for agentcrew.resilience.retry_with_backoff, using a fake sleep_fn
so tests never actually sleep."""

from __future__ import annotations

import pytest

from agentcrew.resilience import retry_with_backoff


class FlakyError(Exception):
    pass


class OtherError(Exception):
    pass


def test_retries_until_success_then_stops():
    calls = {"count": 0}
    sleeps = []

    @retry_with_backoff(exceptions=(FlakyError,), max_attempts=5, sleep_fn=sleeps.append, jitter=False)
    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise FlakyError("not yet")
        return "ok"

    result = flaky()
    assert result == "ok"
    assert calls["count"] == 3
    # Two failures before success => exactly two sleeps.
    assert len(sleeps) == 2


def test_retries_exact_number_of_times_then_raises():
    calls = {"count": 0}
    sleeps = []

    @retry_with_backoff(exceptions=(FlakyError,), max_attempts=3, sleep_fn=sleeps.append, jitter=False)
    def always_fails():
        calls["count"] += 1
        raise FlakyError("boom")

    with pytest.raises(FlakyError):
        always_fails()

    assert calls["count"] == 3
    # Sleeps happen between attempts: max_attempts - 1.
    assert len(sleeps) == 2


def test_does_not_retry_on_excluded_exception_type():
    calls = {"count": 0}
    sleeps = []

    @retry_with_backoff(exceptions=(FlakyError,), max_attempts=5, sleep_fn=sleeps.append, jitter=False)
    def raises_other():
        calls["count"] += 1
        raise OtherError("not retryable")

    with pytest.raises(OtherError):
        raises_other()

    assert calls["count"] == 1
    assert sleeps == []


def test_exponential_backoff_delays_increase():
    sleeps = []

    @retry_with_backoff(
        exceptions=(FlakyError,),
        max_attempts=4,
        base_delay_seconds=1.0,
        max_delay_seconds=100.0,
        sleep_fn=sleeps.append,
        jitter=False,
    )
    def always_fails():
        raise FlakyError("boom")

    with pytest.raises(FlakyError):
        always_fails()

    assert sleeps == [1.0, 2.0, 4.0]


def test_delay_is_capped_at_max_delay_seconds():
    sleeps = []

    @retry_with_backoff(
        exceptions=(FlakyError,),
        max_attempts=6,
        base_delay_seconds=1.0,
        max_delay_seconds=3.0,
        sleep_fn=sleeps.append,
        jitter=False,
    )
    def always_fails():
        raise FlakyError("boom")

    with pytest.raises(FlakyError):
        always_fails()

    assert max(sleeps) <= 3.0
    assert sleeps[-1] == 3.0


def test_jitter_keeps_delay_within_expected_bounds():
    sleeps = []

    @retry_with_backoff(
        exceptions=(FlakyError,),
        max_attempts=2,
        base_delay_seconds=2.0,
        max_delay_seconds=100.0,
        sleep_fn=sleeps.append,
        jitter=True,
    )
    def always_fails():
        raise FlakyError("boom")

    with pytest.raises(FlakyError):
        always_fails()

    assert len(sleeps) == 1
    # base_delay_seconds * 2**0 == 2.0, jittered by [0.5, 1.5)
    assert 1.0 <= sleeps[0] < 3.0
