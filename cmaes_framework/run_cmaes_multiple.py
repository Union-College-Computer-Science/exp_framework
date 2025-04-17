"""
Runs CMA-ES sequentially for multiple instances.
Creates an output folder with a timestamp for the experiment,
and runs the specified number of instances one after another.
"""

import argparse
from run_cmaes import run
from datetime import datetime
import os

def run_cmaes_instance(mode, gens, sigma, output_folder, run_number):
    """
    Wrapper to call the run() function from run_cmaes.py with provided parameters.
    """
    print(f"\n[Run {run_number}] Starting CMA-ES run...")
    run(mode, gens, sigma, output_folder=output_folder, run_number=run_number)
    print(f"[Run {run_number}] Finished CMA-ES run.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequential CMA-ES runner')
    parser.add_argument('--mode', default="h", help='Output mode: h-headless, s-screen, v-video, b-both')
    parser.add_argument('--gens', type=int, default=100, help='Number of generations to run')
    parser.add_argument('--sigma', type=float, default=3, help='Sigma value for CMA-ES')
    parser.add_argument('--runs', type=int, default=5, help='Total number of CMA-ES runs to execute')

    args = parser.parse_args()

    # Create a timestamped output folder for all runs
    timestamp_folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for run_number in range(1, args.runs + 1):
        run_cmaes_instance(args.mode, args.gens, args.sigma, timestamp_folder, run_number)

    print("\n✅ All CMA-ES runs completed.")