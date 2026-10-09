"""
Synthetic dataset generator for the ML pre-screening model.

Ground truth is produced by Banker's Algorithm itself: for a randomly
generated resource-allocation snapshot plus a candidate request, we ask
Banker's Algorithm whether granting the request would leave the system in
a safe state. That safe/unsafe verdict is the label. This guarantees the
labels are correct by construction (no hand-labeling, no guessing) and
keeps the "ML learns to approximate Banker's judgement" framing honest.

Output: results/ml_dataset.csv
Columns: held,requested,total_available,remaining_need,alloc_ratio,
         req_ratio,req_r0..req_rN,avail_r0..avail_rN,label
label: 1 = safe (Banker's would GRANT), 0 = unsafe/risky (Banker's would DEFER)
"""

import csv
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithms.banker import BankerAlgorithm

NUM_RESOURCE_TYPES = 3
NUM_SAMPLES = 4000
SEED = 7


def random_state(rng, num_types):
    """Build a random-but-plausible multi-process resource state.

    Parameters below were empirically tuned (see project notes) so the
    resulting safe/unsafe label distribution is reasonably balanced rather
    than trivially skewed to one class -- an earlier naive version of this
    generator produced >99% unsafe labels because processes were given
    max-demand and initial-allocation ranges that made almost every state
    saturated. A classifier trained on a wildly imbalanced set would just
    learn to always predict the majority class, which is not a meaningful
    "prediction," so this balance matters for the ML results to mean
    anything.
    """
    num_processes = rng.randint(3, 8)
    available = [rng.randint(6, 15) for _ in range(num_types)]
    max_demand = {}
    for pid in range(num_processes):
        max_demand[pid] = [rng.randint(1, max(1, int(available[t] * 0.8))) for t in range(num_types)]

    banker = BankerAlgorithm(list(available), max_demand)

    # Give each process a random partial allocation (<= its max, <= available at time)
    for pid in range(num_processes):
        for t in range(num_types):
            upper = min(max_demand[pid][t], banker.available[t])
            if upper > 0:
                frac = rng.uniform(0.2, 0.5)
                grant = int(upper * frac)
                banker.available[t] -= grant
                banker.allocation[pid][t] += grant
                banker.need[pid][t] -= grant
    return banker, num_processes


def generate(num_samples=NUM_SAMPLES, num_types=NUM_RESOURCE_TYPES, seed=SEED):
    rng = random.Random(seed)
    rows = []

    while len(rows) < num_samples:
        banker, num_processes = random_state(rng, num_types)
        pid = rng.randrange(num_processes)
        need = banker.need[pid]
        if all(n == 0 for n in need):
            continue
        request_vector = [rng.randint(0, need[t]) for t in range(num_types)]
        if all(v == 0 for v in request_vector):
            continue

        held = sum(banker.allocation[pid])
        requested = sum(request_vector)
        total_available = sum(banker.available)
        remaining_need = sum(need)
        max_total = sum(banker.max_demand[pid]) + 1e-9
        alloc_ratio = held / max_total
        req_ratio = requested / (total_available + 1e-9)

        # Evaluate ground truth WITHOUT mutating the shared banker state
        # permanently (request() mutates internally but rolls back on unsafe).
        decision, _ = banker.request(pid, request_vector)
        label = 1 if decision == "GRANTED" else 0
        if decision == "GRANTED":
            banker.release(pid)  # reset for cleanliness; state is discarded anyway

        row = {
            "held": held,
            "requested": requested,
            "total_available": total_available,
            "remaining_need": remaining_need,
            "alloc_ratio": alloc_ratio,
            "req_ratio": req_ratio,
        }
        for t in range(num_types):
            row[f"req_r{t}"] = request_vector[t]
        for t in range(num_types):
            row[f"avail_r{t}"] = banker.available[t]
        row["label"] = label
        rows.append(row)

    return rows


def main():
    rows = generate()
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "ml_dataset.csv")
    out_path = os.path.abspath(out_path)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_safe = sum(r["label"] == 1 for r in rows)
    n_unsafe = len(rows) - n_safe
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Class balance: safe={n_safe} ({n_safe/len(rows):.1%})  unsafe={n_unsafe} ({n_unsafe/len(rows):.1%})")


if __name__ == "__main__":
    main()
