"""Run the full smart-meter attack detection pipeline in order.

Usage (from the project root, with the virtual environment active):
    python run_pipeline.py              # run all steps
    python run_pipeline.py --from models   # start from a given step
    python run_pipeline.py --app        # run all steps, then launch the dashboard
"""
import os
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("load_data", "src/load_data.py"),
    ("attacks", "src/attacks.py"),
    ("features", "src/features.py"),
    ("models", "src/models.py"),
    ("evaluate", "src/evaluate.py"),
]


def run(cmd):
    """Run a command from the project root and stop the pipeline if it fails."""
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    if result.returncode != 0:
        print(f"\nFAILED: {' '.join(cmd)} (exit code {result.returncode})")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run the detection pipeline.")
    parser.add_argument(
        "--from",
        dest="start",
        choices=[name for name, _ in STEPS],
        default=STEPS[0][0],
        help="first step to run (default: load_data)",
    )
    parser.add_argument("--app", action="store_true", help="launch the dashboard afterwards")
    args = parser.parse_args()

    names = [name for name, _ in STEPS]
    steps = STEPS[names.index(args.start):]

    total_start = time.time()
    for name, script in steps:
        print(f"\n=== Step: {name} ({script}) ===")
        step_start = time.time()
        run([sys.executable, script])
        print(f"--- {name} finished in {time.time() - step_start:.1f}s ---")

    print(f"\nPipeline complete in {(time.time() - total_start) / 60:.1f} minutes.")

    if args.app:
        print("\nLaunching dashboard...")
        run([sys.executable, "-m", "streamlit", "run", "app.py"])


if __name__ == "__main__":
    main()