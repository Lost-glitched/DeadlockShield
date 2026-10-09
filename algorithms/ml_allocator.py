"""
ML-assisted resource allocation — PREDICTIVE PRE-SCREENING LAYER ONLY.

Design contract (must not be violated anywhere in this project):
    The trained classifier NEVER makes a final grant decision on its own.
    It only predicts whether a request is likely to be "risky" (i.e.
    likely to lead to an unsafe state). Banker's Algorithm remains the
    sole authority that can actually GRANT a request.

    - If the model predicts "safe" with confidence >= threshold, we still
      require the request to pass Banker's Algorithm before granting —
      but we count this as a case where the model's prediction *agreed*
      with the outcome, and in a real system this is where a full,
      expensive safety re-verification could plausibly be skipped for
      low-risk requests if the deployment context tolerated it. This
      project does NOT skip Banker's check even here; it always runs
      Banker's check and simply RECORDS how often the model's prediction
      would have been correct if we had trusted it, to answer the
      research question empirically without weakening correctness.
    - If the model predicts "unsafe"/risky, the request is always sent to
      Banker's Algorithm as normal.

This keeps the experiment scientifically honest: the ML strategy's
*granted/deferred outcomes are always identical to Banker's Algorithm's*
(same safety guarantee), while we separately measure prediction quality
(accuracy/precision/recall/F1) and the hypothetical safety-check savings
(how many of the full safety checks were, in hindsight, unnecessary
because the model already predicted "safe" correctly).
"""

import pickle
import time
from typing import Dict, List, Tuple

from algorithms.banker import BankerAlgorithm


class MLAssistedAllocator:
    def __init__(self, available: List[int], max_demand: Dict[int, List[int]], model_path: str):
        # Banker's Algorithm instance remains the deterministic safety authority.
        self.banker = BankerAlgorithm(available, max_demand)

        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        self.model = bundle["model"]
        self.feature_names = bundle["feature_names"]

        self.num_types = self.banker.num_types

        # instrumentation
        self.predictions_made = 0
        self.total_prediction_time = 0.0
        self.model_predicted_safe = 0
        self.model_predicted_unsafe = 0
        self.model_correct = 0  # prediction matched Banker's actual outcome
        self.hypothetical_checks_saved = 0  # model said safe AND Banker agreed

    def _build_features(self, pid: int, request_vector: List[int]) -> List[float]:
        allocation = self.banker.allocation[pid]
        need = self.banker.need[pid]
        available = self.banker.available
        total_available = sum(available)
        held = sum(allocation)
        requested = sum(request_vector)
        remaining_need = sum(need)
        num_types = self.num_types

        return [
            held,
            requested,
            total_available,
            remaining_need,
            held / (sum(self.banker.max_demand[pid]) + 1e-9),
            requested / (total_available + 1e-9),
            *([request_vector[i] for i in range(num_types)]),
            *([available[i] for i in range(num_types)]),
        ]

    def request(self, pid: int, request_vector: List[int]) -> Tuple[str, str]:
        features = self._build_features(pid, request_vector)

        t0 = time.perf_counter()
        pred = self.model.predict([features])[0]  # 1 = predicted safe, 0 = predicted unsafe/risky
        self.total_prediction_time += time.perf_counter() - t0
        self.predictions_made += 1

        if pred == 1:
            self.model_predicted_safe += 1
        else:
            self.model_predicted_unsafe += 1

        # Banker's Algorithm ALWAYS runs — it is the safety authority.
        decision, reason = self.banker.request(pid, request_vector)
        actual_safe = 1 if decision == "GRANTED" else 0

        if pred == actual_safe:
            self.model_correct += 1
            if pred == 1:
                self.hypothetical_checks_saved += 1

        return decision, reason

    def release(self, pid: int) -> None:
        self.banker.release(pid)

    def snapshot(self) -> Dict:
        base = self.banker.snapshot()
        base.update(
            {
                "predictions_made": self.predictions_made,
                "total_prediction_time": self.total_prediction_time,
                "model_predicted_safe": self.model_predicted_safe,
                "model_predicted_unsafe": self.model_predicted_unsafe,
                "model_correct": self.model_correct,
                "hypothetical_checks_saved": self.hypothetical_checks_saved,
            }
        )
        return base
