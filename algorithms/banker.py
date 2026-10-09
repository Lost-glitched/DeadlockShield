"""
Banker's Algorithm — deterministic deadlock-avoidance / safety authority.

Role in this project:
    Banker's Algorithm is the ONLY component in this project that makes a
    provable safe/unsafe judgement about a resource-allocation state. It is
    used in two places:
        1. As a standalone strategy (baseline).
        2. As the mandatory verification step behind the ML-assisted
           strategy (ml_allocator.py) whenever the ML model is uncertain
           or predicts risk. ML NEVER bypasses this check when it matters;
           it only decides whether the check is *necessary*.

This is a classic, textbook-style implementation (Available / Max /
Allocation / Need matrices) with instrumentation added so we can measure
how often the safety algorithm actually runs and how long it takes.
"""

import time
from typing import Dict, List, Tuple


class BankerAlgorithm:
    def __init__(self, available: List[int], max_demand: Dict[int, List[int]]):
        """
        available:   list[int], length = num_resource_types
        max_demand:  {process_id: [max_R0, max_R1, ...]}
        """
        self.num_types = len(available)
        self.available = list(available)
        self.max_demand = {pid: list(v) for pid, v in max_demand.items()}
        self.allocation = {pid: [0] * self.num_types for pid in max_demand}
        self.need = {pid: list(v) for pid, v in max_demand.items()}

        # instrumentation
        self.safety_checks_performed = 0
        self.total_safety_check_time = 0.0
        self.unsafe_states_rejected = 0
        self.requests_granted = 0
        self.requests_deferred = 0

    # ------------------------------------------------------------------ #
    def is_safe_state(
        self, available: List[int], allocation: Dict[int, List[int]], need: Dict[int, List[int]]
    ) -> Tuple[bool, List[int]]:
        """Standard safety algorithm. Returns (is_safe, safe_sequence)."""
        work = list(available)
        finish = {pid: False for pid in allocation}
        safe_sequence = []

        progress = True
        while progress:
            progress = False
            for pid in allocation:
                if finish[pid]:
                    continue
                if all(need[pid][i] <= work[i] for i in range(self.num_types)):
                    for i in range(self.num_types):
                        work[i] += allocation[pid][i]
                    finish[pid] = True
                    safe_sequence.append(pid)
                    progress = True

        is_safe = all(finish.values())
        return is_safe, safe_sequence

    # ------------------------------------------------------------------ #
    def request(self, pid: int, request_vector: List[int]) -> Tuple[str, str]:
        """
        Attempt to grant `request_vector` resources to process `pid`.
        Returns (decision, reason) where decision in {"GRANTED", "DEFERRED"}.
        """
        need_pid = self.need[pid]

        if any(request_vector[i] > need_pid[i] for i in range(self.num_types)):
            self.requests_deferred += 1
            return "DEFERRED", "exceeds_declared_max"

        if any(request_vector[i] > self.available[i] for i in range(self.num_types)):
            self.requests_deferred += 1
            return "DEFERRED", "insufficient_available"

        # Tentatively allocate
        for i in range(self.num_types):
            self.available[i] -= request_vector[i]
            self.allocation[pid][i] += request_vector[i]
            self.need[pid][i] -= request_vector[i]

        t0 = time.perf_counter()
        safe, _ = self.is_safe_state(self.available, self.allocation, self.need)
        self.total_safety_check_time += time.perf_counter() - t0
        self.safety_checks_performed += 1

        if safe:
            self.requests_granted += 1
            return "GRANTED", "safe_state"

        # Rollback — granting this request would be unsafe
        for i in range(self.num_types):
            self.available[i] += request_vector[i]
            self.allocation[pid][i] -= request_vector[i]
            self.need[pid][i] += request_vector[i]

        self.unsafe_states_rejected += 1
        self.requests_deferred += 1
        return "DEFERRED", "unsafe_state"

    # ------------------------------------------------------------------ #
    def release(self, pid: int) -> None:
        """Process finished: release all resources it currently holds."""
        for i in range(self.num_types):
            self.available[i] += self.allocation[pid][i]
            self.need[pid][i] += self.allocation[pid][i]
            self.allocation[pid][i] = 0

    def snapshot(self) -> Dict:
        return {
            "available": list(self.available),
            "safety_checks_performed": self.safety_checks_performed,
            "total_safety_check_time": self.total_safety_check_time,
            "unsafe_states_rejected": self.unsafe_states_rejected,
            "requests_granted": self.requests_granted,
            "requests_deferred": self.requests_deferred,
        }
