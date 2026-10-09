# Comparative Analysis of Classical and ML-Assisted Resource Allocation and Deadlock Avoidance in Concurrent Systems

## Overview

This is a small, research-style Python prototype built for an Operating
Systems (Module 4 — Concurrency) case study. It simulates concurrent
processes requesting shared resources under four different control
strategies, driven through real inter-process communication, and measures
their behaviour experimentally.

This is a **controlled simulation/prototype**, not a production system,
and no claim here should be read as "ML replaces OS synchronization" or
"ML guarantees deadlock prevention." ML is used strictly as an optional
predictive pre-screening layer; a classical algorithm (Banker's) remains
the sole safety authority at all times.

## Problem Statement

Classical deadlock-avoidance (Banker's Algorithm) is provably correct but
requires a full safety check on every request. Pure synchronization
primitives (semaphores, Bakery's mutual exclusion) are cheap but have no
notion of a "safe state" and can stall indefinitely under circular wait.
This project asks: **can a lightweight ML classifier meaningfully predict
request risk, and would that prediction — if trusted — have reduced the
number of full safety checks needed, without ever actually removing
Banker's Algorithm as the final authority?** The experiments, not
assumptions, answer this.

## Objectives

1. Implement Banker's Algorithm as the deterministic deadlock-avoidance baseline.
2. Implement a semaphore-based resource manager to demonstrate pure synchronization (no safety check).
3. Implement Lamport's Bakery Algorithm manually for mutual exclusion over a shared allocation table.
4. Implement inter-process communication (IPC) so worker processes and the resource manager run as separate OS processes.
5. Train a small, explainable ML model (Random Forest) to predict request risk, using Banker's Algorithm as ground truth.
6. Run identical, seeded workloads through all four strategies and measure them fairly.
7. Generate graphs and a defensible experimental comparison — with no fabricated numbers.

## Selected OS Concepts (Module 4 syllabus)

* Inter-process communication (`multiprocessing.Queue`, worker processes ↔ central manager)
* Synchronization (semaphores)
* Peterson's/Bakery-family mutual exclusion (Lamport's Bakery Algorithm, implemented manually)
* Semaphores
* Classical synchronization / resource-allocation problems
* Deadlock avoidance (Banker's Algorithm)
* Multiprocessors and locking (thread pool contention for semaphore/Bakery strategies)

## System Architecture

```
Worker Processes (multiprocessing.Process, one per logical process)
        │  request\_vector, timestamps
        ▼
  IPC Request Queue (multiprocessing.Queue)
        ▼
  Central Resource Manager (main process)
        │
        ├─ Banker's Algorithm strategy        (single-threaded, sequential — safety
        │                                       checks must be atomic to be correct)
        ├─ Semaphore-based strategy            (small thread pool — real thread-level
        │                                       contention on shared semaphores)
        ├─ Bakery-based strategy               (small thread pool — real thread-level
        │                                       contention protected by a hand-written
        │                                       Bakery mutual-exclusion lock)
        └─ ML-assisted strategy                (single-threaded; ML predicts risk,
                                                  Banker's Algorithm always verifies)
        ▼
  IPC Response Queue (one per worker, multiprocessing.Queue)
        ▼
Worker Processes receive GRANTED / DEFERRED / TIMEOUT
```

**These four components are NOT competing solutions to the same problem.**
They sit at different layers:

|Component|Layer|Question it answers|
|-|-|-|
|IPC (`multiprocessing.Queue`)|Communication|How do independent processes exchange requests/decisions?|
|Semaphore|Synchronization|How do we limit concurrent access to a bounded count of units?|
|Bakery Algorithm|Mutual exclusion|How do we serialize access to shared state without hardware atomics?|
|Banker's Algorithm|Deadlock avoidance|Is granting this request provably safe?|
|ML model|Predictive pre-screening|Can we predict the Banker's Algorithm outcome cheaply, without replacing it?|

## Algorithms Implemented

* **`algorithms/banker.py`** — classic Available/Max/Allocation/Need Banker's Algorithm with a standard safety algorithm, instrumented to count safety checks and unsafe rejections.
* **`algorithms/semaphore\_manager.py`** — one `threading.Semaphore` per resource type; blocking acquire with a bounded timeout (a timeout is logged as a potential-deadlock event, not hidden).
* **`algorithms/bakery.py`** — a from-scratch array-based Lamport's Bakery lock (ticket numbers + choosing flags), protecting a simple first-fit allocation table. Not a wrapper around `threading.Lock`.
* **`algorithms/ml\_allocator.py`** — wraps Banker's Algorithm; a trained classifier predicts request risk purely for measurement — every request is still verified by Banker's Algorithm, so the ML strategy's actual grant/deny outcomes are always identical to Banker's.

## ML Methodology

1. `ml/generate\_dataset.py` builds random-but-plausible multi-process resource states and asks Banker's Algorithm for the safe/unsafe ground-truth label for a candidate request. Parameters are tuned so the resulting label distribution is roughly balanced (\~48/52), not degenerate.
2. `ml/train\_model.py` does a proper train/test split (75/25, stratified) and trains a `RandomForestClassifier` (`n\_estimators=100, max\_depth=8`) — no deep learning.
3. `ml/evaluate\_model.py` evaluates **only** on the held-out test rows saved during training, so there is no data leakage. Reports accuracy, precision, recall, F1, confusion matrix, false-positive/negative rate, and prediction latency.

## Experimental Methodology

* `simulator/workload.py` generates a deterministic, seeded workload (Max demand per process, Available vector, and a 2–4-chunk incremental request sequence per process — the classic Banker's process lifecycle: request in pieces, release everything on completion).
* **The identical seeded workload is reused across all four strategies** for a given (process count, trial), so no strategy gets an easier or harder workload than another.
* `experiments/run\_experiments.py` sweeps process counts `\[5, 10, 20, 50, 100]` with 3 trials each, running every strategy through real multiprocessing IPC, and writes one row per (strategy, process count, trial) to `results/experiment\_results.csv`.

## Metrics

Only metrics that are actually meaningful for each strategy are used (see report guidance below):

* **All strategies:** execution time, average wait time, throughput, resource utilization, granted/deferred/timed-out counts.
* **Banker's / ML-assisted only:** safety checks performed, unsafe states rejected.
* **Semaphore only:** acquire timeouts (its analogue of "potential deadlock" — it has no safety concept, so it cannot have "unsafe states rejected").
* **ML only:** accuracy, precision, recall, F1, confusion matrix, false-positive/negative rate, prediction latency, hypothetical safety-checks-saved (how often the model's "safe" prediction agreed with Banker's actual grant — a measure of potential overhead reduction, not an actual skipped check, since Banker's always runs here for correctness).

**Do not report a blanket "X is faster than Y."** Semaphore and Bakery numbers reflect pure synchronization overhead; Banker's and ML-assisted numbers reflect safety-checking overhead. They are answering different questions, even when plotted on the same axis for convenience.

## Installation (Windows PowerShell / CMD)

```powershell
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
```

Recommended Python version: 3.10–3.12.

## How to Run

Run everything in order with one command:

```powershell
python main.py
```

Or run each stage individually:

```powershell
python -m ml.generate\_dataset
python -m ml.train\_model
python -m ml.evaluate\_model
python -m experiments.run\_experiments
python -m experiments.generate\_plots
```

* CSV results appear in `results/`.
* Trained model appears in `models/model.pkl`.
* Graphs appear in `plots/`.

To verify it worked: `results/experiment\_results.csv` should have
`4 strategies × 5 process counts × 3 trials = 60` rows, and `plots/`
should contain 9 PNG files.

## Project Structure

```
os\_project/
├── README.md
├── requirements.txt
├── main.py
├── algorithms/
│   ├── banker.py
│   ├── semaphore\_manager.py
│   ├── bakery.py
│   └── ml\_allocator.py
├── simulator/
│   ├── workload.py
│   ├── process\_worker.py
│   ├── resource\_manager.py
│   └── metrics.py
├── ml/
│   ├── generate\_dataset.py
│   ├── train\_model.py
│   └── evaluate\_model.py
├── experiments/
│   ├── run\_experiments.py
│   └── generate\_plots.py
├── results/    (CSV output — generated, not committed with fake data)
├── plots/      (PNG output — generated, not committed with fake data)
└── models/     (trained model.pkl — generated)
```

## Limitations

* Python's GIL means `threading`-based strategies (semaphore, Bakery) do not achieve true multicore parallel execution; timing results reflect real OS-level scheduling and contention, not multicore speed-up.
* The semaphore strategy's acquire timeout (2 seconds by default, `algorithms/semaphore\_manager.py`) is a simulation safety valve, not a real-world deadlock-detection mechanism — at higher process counts it can dominate execution-time results. This is a known, documented limitation.
* Deadlock *recovery* and *rollback* are explicitly out of scope; this project covers avoidance (Banker's) and the practical consequence of not having avoidance (semaphore/Bakery timeouts and deferrals).
* The ML model's usefulness is scoped strictly to *predicting* Banker's outcome; it never substitutes for it.

## Future Work

* Extend IPC to use `multiprocessing.Pipe` for a point-to-point comparison against the current queue-based design.
* Explore a true deadlock-detection algorithm (resource-allocation graph cycle detection) as a fifth strategy.
* Experiment with feature importance analysis on the trained Random Forest to explain which request features drive its risk predictions.
* ## 

