"""
Deterministic, seeded workload generator.

The SAME workload (same Max demand per process, same Available vector,
same per-process request-chunk sequence) is generated for a given
(num_processes, num_resource_types, seed) and reused across every
strategy, so comparisons are fair. Nothing here depends on wall-clock
time or hardware — reproducibility only depends on the seed.

Process model (classic Banker's-style lifecycle):
    Each process i declares Max[i] up front. It then requests its
    resources in 2-4 incremental chunks (not all at once), and only
    releases everything when it has received its full Max allocation
    (i.e. when it "finishes"). This is what makes Banker's Algorithm's
    safety check meaningful: partial holdings + pending future need is
    exactly the scenario deadlock avoidance is designed for.
"""

import random
from typing import Dict, List, Tuple


class Workload:
    def __init__(
        self,
        num_processes: int,
        num_resource_types: int,
        available: List[int],
        max_demand: Dict[int, List[int]],
        request_chunks: Dict[int, List[List[int]]],
    ):
        self.num_processes = num_processes
        self.num_resource_types = num_resource_types
        self.available = available
        self.max_demand = max_demand
        self.request_chunks = request_chunks  # pid -> ordered list of request vectors


def generate_workload(num_processes: int, num_resource_types: int = 3, seed: int = 42) -> Workload:
    rng = random.Random(seed)

    # Total instances per resource type scale mildly with process count so the
    # system stays under realistic pressure at every scale (not trivially
    # abundant, not trivially starved).
    available = [max(10, int(num_processes * rng.uniform(0.6, 0.9))) for _ in range(num_resource_types)]

    max_demand: Dict[int, List[int]] = {}
    request_chunks: Dict[int, List[List[int]]] = {}

    for pid in range(num_processes):
        # Each process's max demand per type is bounded by a fraction of total
        # availability so no single process can trivially exceed the system.
        max_vec = [rng.randint(1, max(1, available[t] // 2)) for t in range(num_resource_types)]
        max_demand[pid] = max_vec

        # Split max demand into 2-4 incremental request chunks per resource type.
        num_chunks = rng.randint(2, 4)
        chunks = [[0] * num_resource_types for _ in range(num_chunks)]
        for t in range(num_resource_types):
            remaining = max_vec[t]
            for c in range(num_chunks - 1):
                if remaining <= 0:
                    break
                take = rng.randint(0, remaining)
                chunks[c][t] = take
                remaining -= take
            chunks[num_chunks - 1][t] += remaining
        request_chunks[pid] = chunks

    return Workload(num_processes, num_resource_types, available, max_demand, request_chunks)
