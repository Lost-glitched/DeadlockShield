"""Lightweight metrics collector used by the resource manager during a run."""

import statistics
from typing import Dict, List


class MetricsCollector:
    def __init__(self):
        self.wait_times: List[float] = []
        self.utilization_samples: List[float] = []
        self.granted = 0
        self.deferred = 0
        self.timed_out = 0
        self.start_time = None
        self.end_time = None

    def record_utilization(self, sample: float) -> None:
        self.utilization_samples.append(sample)

    def record_request(self, wait_time: float, decision: str) -> None:
        self.wait_times.append(wait_time)
        if decision == "GRANTED":
            self.granted += 1
        elif decision == "DEFERRED":
            self.deferred += 1
        elif decision == "TIMEOUT":
            self.timed_out += 1

    def start(self) -> None:
        import time
        self.start_time = time.perf_counter()

    def stop(self) -> None:
        import time
        self.end_time = time.perf_counter()

    def summary(self) -> Dict:
        execution_time = (self.end_time - self.start_time) if self.start_time and self.end_time else 0.0
        total_requests = self.granted + self.deferred + self.timed_out
        throughput = self.granted / execution_time if execution_time > 0 else 0.0
        return {
            "execution_time_sec": execution_time,
            "total_requests": total_requests,
            "requests_granted": self.granted,
            "requests_deferred": self.deferred,
            "requests_timed_out": self.timed_out,
            "avg_wait_time_sec": statistics.mean(self.wait_times) if self.wait_times else 0.0,
            "max_wait_time_sec": max(self.wait_times) if self.wait_times else 0.0,
            "throughput_grants_per_sec": throughput,
            "avg_resource_utilization": (
                statistics.mean(self.utilization_samples) if self.utilization_samples else 0.0
            ),
        }
