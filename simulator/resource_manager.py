"""
Central Resource Manager — IPC consumer side.

    Worker Processes --(multiprocessing.Queue)--> Resource Manager --> Strategy

The manager lives in the main process. Worker processes (spawned with
`multiprocessing.Process`) send REQUEST/RELEASE messages over a shared
`multiprocessing.Queue`; the manager replies on a private per-process
response queue. This is the IPC component required by the syllabus
("IPC in Unix" — realized here with Python's cross-platform
multiprocessing.Queue, which is a message-queue/pipe-based IPC
mechanism).

Two consumption modes, chosen per strategy for a principled reason
(see algorithms/*.py docstrings):

    * Banker's Algorithm and ML-assisted allocator are executed
      SEQUENTIALLY in the manager's main loop. Banker's safety check is,
      by definition, an atomic global decision — that is what makes it
      correct. There is nothing to "synchronize" here because there is
      only one decision-maker.

    * Semaphore-based and Bakery-based strategies are executed by a small
      pool of manager-side worker THREADS pulling concurrently from the
      same IPC queue. This is where real synchronization/mutual-exclusion
      contention actually occurs and is measured.
"""

import multiprocessing as mp
import threading
import time
from typing import Dict, List

from algorithms.banker import BankerAlgorithm
from algorithms.semaphore_manager import SemaphoreResourceManager
from algorithms.bakery import BakeryResourceManager
from algorithms.ml_allocator import MLAssistedAllocator
from simulator.metrics import MetricsCollector
from simulator.process_worker import worker_process

THREAD_POOL_SIZE_CAP = 8


def _spawn_workers(workload, num_processes):
    request_queue = mp.Queue()
    response_queues = {pid: mp.Queue() for pid in range(num_processes)}
    workers = []
    pool_size = min(THREAD_POOL_SIZE_CAP, num_processes)
    for pid in range(num_processes):
        thread_slot = pid % pool_size
        p = mp.Process(
            target=worker_process,
            args=(pid, workload.request_chunks[pid], request_queue, response_queues[pid], thread_slot),
        )
        workers.append(p)
    return request_queue, response_queues, workers, pool_size


def _drain_and_join(workers, request_queue):
    for p in workers:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
    try:
        while True:
            request_queue.get_nowait()
    except Exception:
        pass


# ---------------------------------------------------------------------- #
def run_banker(workload, num_processes: int) -> Dict:
    request_queue, response_queues, workers, _ = _spawn_workers(workload, num_processes)
    strategy = BankerAlgorithm(workload.available, workload.max_demand)
    metrics = MetricsCollector()
    total_capacity = sum(workload.available)

    for p in workers:
        p.start()

    metrics.start()
    finished = 0
    while finished < num_processes:
        msg = request_queue.get()
        if msg["type"] == "RELEASE":
            strategy.release(msg["pid"])
            finished += 1
            continue
        decision, reason = strategy.request(msg["pid"], msg["vector"])
        wait = time.perf_counter() - msg["send_time"]
        metrics.record_request(wait, decision)
        held = sum(sum(v) for v in strategy.allocation.values())
        metrics.record_utilization(held / total_capacity if total_capacity else 0.0)
        response_queues[msg["pid"]].put({"decision": decision, "reason": reason})
    metrics.stop()

    _drain_and_join(workers, request_queue)

    result = metrics.summary()
    result.update({"strategy": "banker"})
    result.update({f"banker_{k}": v for k, v in strategy.snapshot().items()})
    return result


# ---------------------------------------------------------------------- #
def run_semaphore(workload, num_processes: int) -> Dict:
    request_queue, response_queues, workers, pool_size = _spawn_workers(workload, num_processes)
    strategy = SemaphoreResourceManager(workload.available)
    metrics = MetricsCollector()
    metrics_lock = threading.Lock()
    total_capacity = sum(workload.available)

    finished_count = [0]
    finished_lock = threading.Lock()
    stop_event = threading.Event()

    def pool_worker():
        while not stop_event.is_set():
            try:
                msg = request_queue.get(timeout=0.5)
            except Exception:
                if stop_event.is_set():
                    return
                continue
            if msg["type"] == "RELEASE":
                strategy.release(msg["pid"])
                with finished_lock:
                    finished_count[0] += 1
                    if finished_count[0] >= num_processes:
                        stop_event.set()
                continue
            decision, reason = strategy.request(msg["pid"], msg["vector"])
            wait = time.perf_counter() - msg["send_time"]
            with metrics_lock:
                metrics.record_request(wait, decision)
                held = sum(sum(v) for v in strategy.allocation.values())
                metrics.record_utilization(held / total_capacity if total_capacity else 0.0)
            response_queues[msg["pid"]].put({"decision": decision, "reason": reason})

    threads = [threading.Thread(target=pool_worker) for _ in range(pool_size)]

    for p in workers:
        p.start()
    metrics.start()
    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=60)
    metrics.stop()

    _drain_and_join(workers, request_queue)

    result = metrics.summary()
    result.update({"strategy": "semaphore"})
    result.update({f"semaphore_{k}": v for k, v in strategy.snapshot().items()})
    return result


# ---------------------------------------------------------------------- #
def run_bakery(workload, num_processes: int) -> Dict:
    request_queue, response_queues, workers, pool_size = _spawn_workers(workload, num_processes)
    strategy = BakeryResourceManager(workload.available, pool_size)
    metrics = MetricsCollector()
    metrics_lock = threading.Lock()
    total_capacity = sum(workload.available)

    finished_count = [0]
    finished_lock = threading.Lock()
    stop_event = threading.Event()

    def pool_worker():
        while not stop_event.is_set():
            try:
                msg = request_queue.get(timeout=0.5)
            except Exception:
                if stop_event.is_set():
                    return
                continue
            if msg["type"] == "RELEASE":
                strategy.release(msg["pid"], msg["thread_slot"])
                with finished_lock:
                    finished_count[0] += 1
                    if finished_count[0] >= num_processes:
                        stop_event.set()
                continue
            decision, reason = strategy.request(msg["pid"], msg["vector"], msg["thread_slot"])
            wait = time.perf_counter() - msg["send_time"]
            with metrics_lock:
                metrics.record_request(wait, decision)
                held = sum(sum(v) for v in strategy.allocation.values())
                metrics.record_utilization(held / total_capacity if total_capacity else 0.0)
            response_queues[msg["pid"]].put({"decision": decision, "reason": reason})

    threads = [threading.Thread(target=pool_worker) for _ in range(pool_size)]

    for p in workers:
        p.start()
    metrics.start()
    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=60)
    metrics.stop()

    _drain_and_join(workers, request_queue)

    result = metrics.summary()
    result.update({"strategy": "bakery"})
    result.update({f"bakery_{k}": v for k, v in strategy.snapshot().items()})
    return result


# ---------------------------------------------------------------------- #
def run_ml_assisted(workload, num_processes: int, model_path: str) -> Dict:
    request_queue, response_queues, workers, _ = _spawn_workers(workload, num_processes)
    strategy = MLAssistedAllocator(workload.available, workload.max_demand, model_path)
    metrics = MetricsCollector()
    total_capacity = sum(workload.available)

    for p in workers:
        p.start()

    metrics.start()
    finished = 0
    while finished < num_processes:
        msg = request_queue.get()
        if msg["type"] == "RELEASE":
            strategy.release(msg["pid"])
            finished += 1
            continue
        decision, reason = strategy.request(msg["pid"], msg["vector"])
        wait = time.perf_counter() - msg["send_time"]
        metrics.record_request(wait, decision)
        held = sum(sum(v) for v in strategy.banker.allocation.values())
        metrics.record_utilization(held / total_capacity if total_capacity else 0.0)
        response_queues[msg["pid"]].put({"decision": decision, "reason": reason})
    metrics.stop()

    _drain_and_join(workers, request_queue)

    result = metrics.summary()
    result.update({"strategy": "ml_assisted"})
    result.update({f"ml_{k}": v for k, v in strategy.snapshot().items()})
    return result
