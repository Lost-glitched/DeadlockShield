"""
Lamport's Bakery Algorithm — manual implementation, mutual exclusion ONLY.

Important conceptual note (do not blur this in the report):
    Bakery provides mutual exclusion over a critical section using only
    read/write of shared memory (no hardware atomic instructions), based
    on a "take a ticket number" analogy. It says NOTHING about resource
    allocation policy or deadlock avoidance. Here it protects the critical
    section in which a simple FIRST-FIT allocation decision is made and
    the shared allocation table is mutated. The allocation *policy* itself
    (first-fit, no safety check) is intentionally as naive as the
    semaphore strategy's policy — the only thing being compared against
    the semaphore approach is the *mutual-exclusion mechanism*
    (Bakery vs. threading.Lock), not the allocation policy.

Threads calling lock()/unlock() must pass a unique, stable thread index
(0..N-1). This is a textbook array-based Bakery implementation (Lamport,
1974), not a wrapper around threading.Lock.
"""

import threading
import time
from typing import Dict, List, Tuple


class BakeryLock:
    def __init__(self, num_threads: int):
        self.n = num_threads
        self.choosing = [False] * num_threads
        self.number = [0] * num_threads
        self._array_guard = threading.Lock()  # protects Python-level list writes only;
        # does NOT provide the mutual exclusion itself — the bakery ticket
        # comparison below is what provides that.

    def lock(self, tid: int) -> None:
        with self._array_guard:
            self.choosing[tid] = True
        with self._array_guard:
            self.number[tid] = 1 + max(self.number) if self.number else 1
        with self._array_guard:
            self.choosing[tid] = False

        for other in range(self.n):
            if other == tid:
                continue
            while self.choosing[other]:
                time.sleep(0)  # yield, wait for other to finish choosing a ticket
            while self.number[other] != 0 and (
                (self.number[other], other) < (self.number[tid], tid)
            ):
                time.sleep(0)  # busy-wait: someone with an earlier ticket goes first

    def unlock(self, tid: int) -> None:
        with self._array_guard:
            self.number[tid] = 0


class BakeryResourceManager:
    """First-fit allocator whose shared table is protected by BakeryLock."""

    def __init__(self, available: List[int], num_worker_threads: int):
        self.num_types = len(available)
        self.available = list(available)
        self.allocation: Dict[int, List[int]] = {}
        self.lock = BakeryLock(num_worker_threads)

        self.requests_granted = 0
        self.requests_deferred = 0

    def request(self, pid: int, request_vector: List[int], thread_slot: int) -> Tuple[str, str]:
        self.lock.lock(thread_slot)
        try:
            if all(request_vector[i] <= self.available[i] for i in range(self.num_types)):
                for i in range(self.num_types):
                    self.available[i] -= request_vector[i]
                self.allocation.setdefault(pid, [0] * self.num_types)
                for i in range(self.num_types):
                    self.allocation[pid][i] += request_vector[i]
                self.requests_granted += 1
                decision = ("GRANTED", "first_fit_available")
            else:
                self.requests_deferred += 1
                decision = ("DEFERRED", "insufficient_available")
        finally:
            self.lock.unlock(thread_slot)
        return decision

    def release(self, pid: int, thread_slot: int) -> None:
        self.lock.lock(thread_slot)
        try:
            held = self.allocation.get(pid, [0] * self.num_types)
            self.allocation[pid] = [0] * self.num_types
            for i in range(self.num_types):
                self.available[i] += held[i]
        finally:
            self.lock.unlock(thread_slot)

    def snapshot(self) -> Dict:
        return {
            "requests_granted": self.requests_granted,
            "requests_deferred": self.requests_deferred,
        }
