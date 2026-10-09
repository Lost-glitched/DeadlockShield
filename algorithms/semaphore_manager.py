"""
Semaphore-based resource management — SYNCHRONIZATION ONLY.

Important conceptual note (do not blur this in the report):
    A counting semaphore per resource type enforces mutual exclusion over
    a *count* of available units. It guarantees a process never acquires
    more units than exist, and it makes waiters block until units are
    free. It has NO notion of "safe state" and CANNOT avoid deadlock:
    if two processes each hold some resources and each wait on what the
    other holds, this strategy will hang forever (circular wait).

    To keep the simulation runnable and to actually *observe* this failure
    mode instead of hanging the whole experiment, each acquisition attempt
    uses a bounded timeout. A timeout is logged as a "potential deadlock /
    timeout" event — this is the expected, meaningful result for this
    strategy, not a bug to hide.

This class is driven by a thread pool inside resource_manager.py so that
real multi-threaded contention occurs over the shared semaphore state.
"""

import threading
import time
from typing import Dict, List, Tuple


class SemaphoreResourceManager:
    def __init__(self, available: List[int], acquire_timeout: float = 2.0):
        self.num_types = len(available)
        self.acquire_timeout = acquire_timeout
        self._semaphores = [threading.Semaphore(available[i]) for i in range(self.num_types)]
        self._alloc_lock = threading.Lock()
        self.allocation: Dict[int, List[int]] = {}

        # instrumentation (thread-safe increments via _alloc_lock)
        self.requests_granted = 0
        self.requests_timed_out = 0

    # ------------------------------------------------------------------ #
    def request(self, pid: int, request_vector: List[int]) -> Tuple[str, str]:
        acquired = []
        for i, amount in enumerate(request_vector):
            ok = True
            for _ in range(amount):
                got = self._semaphores[i].acquire(timeout=self.acquire_timeout)
                if not got:
                    ok = False
                    break
                acquired.append(i)
            if not ok:
                # release whatever we grabbed before failing -> avoid permanent leak
                for idx in acquired:
                    self._semaphores[idx].release()
                with self._alloc_lock:
                    self.requests_timed_out += 1
                return "TIMEOUT", "acquire_timeout_possible_deadlock"

        with self._alloc_lock:
            self.allocation.setdefault(pid, [0] * self.num_types)
            for i, amount in enumerate(request_vector):
                self.allocation[pid][i] += amount
            self.requests_granted += 1

        return "GRANTED", "acquired"

    # ------------------------------------------------------------------ #
    def release(self, pid: int) -> None:
        with self._alloc_lock:
            held = self.allocation.get(pid, [0] * self.num_types)
            self.allocation[pid] = [0] * self.num_types
        for i, amount in enumerate(held):
            for _ in range(amount):
                self._semaphores[i].release()

    def snapshot(self) -> Dict:
        return {
            "requests_granted": self.requests_granted,
            "requests_timed_out": self.requests_timed_out,
        }
