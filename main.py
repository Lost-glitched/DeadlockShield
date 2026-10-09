"""
Single entrypoint that runs the whole pipeline in order:
    1. Generate ML training dataset (labeled via Banker's Algorithm)
    2. Train the ML model
    3. Evaluate the ML model on held-out test data
    4. Run the full experiment matrix (all 4 strategies x all process counts)
    5. Generate all graphs

Run with:  python main.py
(You can also run each stage individually — see README.md.)
"""

import subprocess
import sys


STEPS = [
    ("Generating ML dataset", [sys.executable, "-m", "ml.generate_dataset"]),
    ("Training ML model", [sys.executable, "-m", "ml.train_model"]),
    ("Evaluating ML model", [sys.executable, "-m", "ml.evaluate_model"]),
    ("Running experiments", [sys.executable, "-m", "experiments.run_experiments"]),
    ("Generating plots", [sys.executable, "-m", "experiments.generate_plots"]),
]


def main():
    for label, cmd in STEPS:
        print(f"\n{'='*70}\n{label}\n{'='*70}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nStep failed: {label} (exit code {result.returncode}). Stopping.")
            sys.exit(result.returncode)
    print("\nAll steps complete. Check results/ and plots/ directories.")


if __name__ == "__main__":
    main()
