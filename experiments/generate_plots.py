"""
Generates all report graphs strictly from results/experiment_results.csv and
results/ml_confusion_matrix.csv / ml_evaluation_metrics.csv. If a CSV file is
missing, that graph is skipped with a printed warning rather than faked.

Run with:  python -m experiments.generate_plots
"""

import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(BASE, "results")
PLOTS = os.path.join(BASE, "plots")

EXP_CSV = os.path.join(RESULTS, "experiment_results.csv")
CONFMAT_CSV = os.path.join(RESULTS, "ml_confusion_matrix.csv")
ML_METRICS_CSV = os.path.join(RESULTS, "ml_evaluation_metrics.csv")

STRATEGY_COLORS = {
    "banker": "#1f77b4",
    "semaphore": "#ff7f0e",
    "bakery": "#2ca02c",
    "ml_assisted": "#d62728",
}
STRATEGY_LABELS = {
    "banker": "Banker's Algorithm",
    "semaphore": "Semaphore-based",
    "bakery": "Bakery-based",
    "ml_assisted": "ML-assisted + Banker verification",
}


def load_experiment_rows():
    if not os.path.exists(EXP_CSV):
        print(f"WARNING: {EXP_CSV} not found — run experiments first. Skipping experiment plots.")
        return []
    with open(EXP_CSV, newline="") as f:
        return list(csv.DictReader(f))


def group_avg(rows, strategy, process_count, field):
    vals = [
        float(r[field])
        for r in rows
        if r["strategy"] == strategy and int(r["num_processes"]) == process_count and r.get(field) not in (None, "")
    ]
    return sum(vals) / len(vals) if vals else None


def plot_metric_vs_processes(rows, field, ylabel, title, filename, strategies=None):
    strategies = strategies or ["banker", "semaphore", "bakery", "ml_assisted"]
    process_counts = sorted({int(r["num_processes"]) for r in rows})
    if not process_counts:
        return

    plt.figure(figsize=(7, 5))
    plotted = False
    for strat in strategies:
        ys = [group_avg(rows, strat, n, field) for n in process_counts]
        if all(y is None for y in ys):
            continue
        xs = [n for n, y in zip(process_counts, ys) if y is not None]
        ys = [y for y in ys if y is not None]
        plt.plot(xs, ys, marker="o", label=STRATEGY_LABELS.get(strat, strat), color=STRATEGY_COLORS.get(strat))
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel("Number of Processes")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOTS, filename)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_bar_by_strategy(rows, field, ylabel, title, filename, strategies=None):
    strategies = strategies or ["banker", "semaphore", "bakery", "ml_assisted"]
    values = []
    labels = []
    for strat in strategies:
        vals = [float(r[field]) for r in rows if r["strategy"] == strat and r.get(field) not in (None, "")]
        if not vals:
            continue
        values.append(sum(vals) / len(vals))
        labels.append(STRATEGY_LABELS.get(strat, strat))
    if not values:
        return
    plt.figure(figsize=(7, 5))
    plt.bar(labels, values, color=[STRATEGY_COLORS.get(s, "#999") for s in strategies if s in [k for k in strategies]][: len(values)])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out = os.path.join(PLOTS, filename)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_confusion_matrix():
    if not os.path.exists(CONFMAT_CSV):
        print(f"WARNING: {CONFMAT_CSV} not found — skipping confusion matrix plot.")
        return
    with open(CONFMAT_CSV, newline="") as f:
        reader = list(csv.reader(f))
    matrix = [[int(reader[1][1]), int(reader[1][2])], [int(reader[2][1]), int(reader[2][2])]]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Unsafe", "Predicted Safe"])
    ax.set_yticklabels(["Actual Unsafe", "Actual Safe"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center",
                     color="white" if matrix[i][j] > (max(max(matrix)) / 2) else "black")
    ax.set_title("ML Model Confusion Matrix (held-out test set)")
    fig.colorbar(im)
    plt.tight_layout()
    out = os.path.join(PLOTS, "ml_confusion_matrix.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def plot_ml_classification_metrics():
    if not os.path.exists(ML_METRICS_CSV):
        print(f"WARNING: {ML_METRICS_CSV} not found — skipping ML metrics plot.")
        return
    metrics_of_interest = ["accuracy", "precision", "recall", "f1_score"]
    values = {}
    with open(ML_METRICS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["metric"] in metrics_of_interest:
                values[row["metric"]] = float(row["value"])
    if not values:
        return
    plt.figure(figsize=(6, 5))
    plt.bar(list(values.keys()), list(values.values()), color="#6a3d9a")
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("ML Model Classification Performance (held-out test set)")
    plt.tight_layout()
    out = os.path.join(PLOTS, "ml_classification_metrics.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def main():
    os.makedirs(PLOTS, exist_ok=True)
    rows = load_experiment_rows()

    if rows:
        plot_metric_vs_processes(
            rows, "execution_time_sec", "Execution Time (sec)",
            "Process Count vs Execution Time", "01_process_count_vs_execution_time.png"
        )
        plot_metric_vs_processes(
            rows, "avg_wait_time_sec", "Average Wait Time (sec)",
            "Process Count vs Average Wait Time", "02_process_count_vs_avg_wait_time.png"
        )
        plot_metric_vs_processes(
            rows, "throughput_grants_per_sec", "Throughput (grants/sec)",
            "Process Count vs Throughput", "03_process_count_vs_throughput.png"
        )
        plot_metric_vs_processes(
            rows, "avg_resource_utilization", "Average Resource Utilization (fraction)",
            "Process Count vs Resource Utilization", "04_process_count_vs_resource_utilization.png"
        )
        plot_bar_by_strategy(
            rows, "avg_wait_time_sec", "Average Allocation Latency / Wait Time (sec)",
            "Algorithm vs Average Allocation Latency", "05_algorithm_vs_allocation_latency.png"
        )
        plot_bar_by_strategy(
            rows, "requests_deferred", "Avg. Deferred/Unsafe Requests per Run",
            "Algorithm vs Deferred / Unsafe Requests", "06_algorithm_vs_deferred_requests.png",
            strategies=["banker", "bakery", "ml_assisted"],  # semaphore uses timeout, not "deferred"
        )
        plot_metric_vs_processes(
            rows, "throughput_grants_per_sec", "Throughput (grants/sec)",
            "Scalability Comparison Across Strategies", "08_scalability_comparison.png"
        )
    else:
        print("No experiment_results.csv rows found — experiment plots skipped.")

    plot_confusion_matrix()
    plot_ml_classification_metrics()

    print("\nPlot generation complete. Check the plots/ directory.")


if __name__ == "__main__":
    main()
