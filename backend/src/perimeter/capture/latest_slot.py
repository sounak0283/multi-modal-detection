"""Single-slot drop-to-latest frame handoff (PLAN.md section 4).

The most important structural decision in the runtime.

A `queue.Queue` between decode and inference fills whenever inference lags, and once
full the pipeline drifts permanently behind live - producing an alert about something
that happened forty seconds ago. For a security product that is not a performance
problem, it is a correctness problem.

A single overwritable slot always yields the freshest frame and silently discards
everything else. Latency stays bounded by one inference pass no matter how far behind
the consumer falls.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class SlotStats:
    published: int
    consumed: int
    dropped: int

    @property
    def drop_rate(self) -> float:
        return self.dropped / self.published if self.published else 0.0


class LatestSlot(Generic[T]):
    """A one-deep mailbox: publishing overwrites, consuming clears.

    Dropped-frame counts are exposed because a high drop rate is the signal that
    inference cannot keep up - the number Phase 1's benchmark exists to establish.

    `blocking=True` is the one deliberate exception to "publish never blocks", and it
    exists for exactly one caller: the Video test page processing an uploaded file with
    `realtime=False`. There, a finite, already-recorded file has no "live" to fall behind
    - completeness matters more than latency, so publish waits for the previous frame to
    be consumed instead of silently discarding whatever inference did not get to in time
    (measured live: with drop-to-latest, adding more modules increased the drop rate
    enough to intermittently lose short, real events instead of just slowing down).
    """

    def __init__(self, *, blocking: bool = False) -> None:
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._item: T | None = None
        self._closed = False
        self._published = 0
        self._consumed = 0
        self._dropped = 0
        self._blocking = blocking

    def publish(self, item: T) -> None:
        """Overwrite the slot. Never blocks the producer - unless constructed with
        `blocking=True`, where it waits for the slot to be free instead of dropping."""
        with self._not_empty:
            if self._blocking:
                while self._item is not None and not self._closed:
                    self._not_empty.wait()
                if self._closed:
                    return
            elif self._item is not None:
                self._dropped += 1
            self._item = item
            self._published += 1
            self._not_empty.notify_all()

    def get(self, timeout: float | None = None) -> T | None:
        """Take the freshest item, waiting up to `timeout`.

        Returns None on timeout or once the slot is closed and drained.
        """
        with self._not_empty:
            if self._item is None and not self._closed:
                self._not_empty.wait(timeout)
            if self._item is None:
                return None
            item, self._item = self._item, None
            self._consumed += 1
            if self._blocking:
                self._not_empty.notify_all()  # wake a publisher waiting for the slot to free up
            return item

    def get_nowait(self) -> T | None:
        with self._not_empty:
            if self._item is None:
                return None
            item, self._item = self._item, None
            self._consumed += 1
            if self._blocking:
                self._not_empty.notify_all()
            return item

    def close(self) -> None:
        """Wake every waiter so consumers can exit promptly on shutdown."""
        with self._not_empty:
            self._closed = True
            self._not_empty.notify_all()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def stats(self) -> SlotStats:
        with self._lock:
            return SlotStats(self._published, self._consumed, self._dropped)
