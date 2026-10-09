"""
Runs the full experimental matrix: for each process count, generate ONE
seeded workload and replay it against all four strategies, repeating for
several trials. Writes results/experiment_results.csv.

This file must be run with:   python -m experiments.run_experiments
(so that the `algorithms`/`simulator` packages resolve correctly.)

IMPORTANT: this script does not fabricate anything — every row in the
output CSV comes from an actual simulated run using real multiprocessing
IPC and the real strategy implementations.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulator.workload import generate_workload
from simulator.resource_manager import run_banker, run_semaphore, run_bakery, run_ml_assisted

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE, "models", "model.pkl")
OUT_PATH = os.path.join(BASE, "results", "experiment_results.csv")

PROCESS_COUNTS = [5, 10, 20, 50, 100]
NUM_RESOURCE_TYPES = 3
TRIALS_PER_CONFIG = 3
WORKLOAD_SEED_BASE = 1000


def run_all(process_counts=PROCESS_COUNTS, trials=TRIALS_PER_CONFIG):
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            "models/model.pkl not found. Run ml/generate_dataset.py and "
            "ml/train_model.py before running experiments (see README)."
        )

    all_rows = []
    for n in process_counts:
        for trial in range(trials):
            seed = WORKLOAD_SEED_BASE + n * 100 + trial
            # Same workload (same seed) reused for every strategy this trial.
            for strategy_name, run_fn in [
                ("banker", run_banker),
                ("semaphore", run_semaphore),
                ("bakery", run_bakery),
                ("ml_assisted", None),  # handled separately below (needs model path)
            ]:
                workload = generate_workload(n, NUM_RESOURCE_TYPES, seed=seed)
                print(f"[n={n} trial={trial}] running {strategy_name} ...")
                if strategy_name == "ml_assisted":
                    result = run_ml_assisted(workload, n, MODEL_PATH)
                else:
                    result = run_fn(workload, n)
                result["num_processes"] = n
                result["trial"] = trial
                result["seed"] = seed
                all_rows.append(result)

    # Union of all keys across strategies (different strategies log different extra fields)
    fieldnames = sorted({k for row in all_rows for k in row.keys()})
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    run_all()
