"""
Runs the latest DiffTaichi experiment. Also launches best_individual to show the best individual so far
and plots fitness over generations after the run is over. It handles:
- Parameter configuration via command-line arguments
- Single or multiple optimization runs
- Real-time visualization of best individuals
- Fitness progression plotting
- Output organization with timestamped/custom experiment directories

By Thomas Breimer
March 6th, 2025

Modified by Takumi Kojima
June 9th, 2025
"""

import argparse
import multiprocessing
import time
import os
from datetime import datetime

import run_cmaes_difftaichi
import best_individual_latest_difftaichi
import plot_fitness_difftaichi


def main():
    parser = argparse.ArgumentParser(
        description="Run a full evolutionary experiment with DiffTaichi.")

    parser.add_argument(
        '--mode',
        choices=["h", "s", "v", "b"],
        default="h",
        help='Execution mode for visualization (h=headless, s=screen, v=video, b=both)')
    parser.add_argument('--gens',
                        type=int,
                        default=100,
                        help='Number of generations to run')
    parser.add_argument('--sigma',
                        type=float,
                        default=1.0,
                        help='Sigma value (exploration strength) for CMA-ES')
    parser.add_argument('--runs',
                        type=int,
                        default=1,
                        help='Number of independent CMA-ES runs')
    parser.add_argument('--hidden_sizes',
                        type=int,
                        nargs='+',
                        default=[10],
                        help='List of hidden layer sizes for the SNN controller')

    # Output params
    parser.add_argument(
        '--exp_name',
        type=str,
        default=None,
        help='Optional experiment folder name (otherwise timestamped)')
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Whether to plot fitness over generations after running')
    parser.add_argument(
        '--visualize_best',
        action='store_true',
        help='Whether to visualize the best individual live during training')
    parser.add_argument(
        '--logs',
        type=str,
        default="false",
        help='Whether to generate SNN logs during visualization (true/false)')

    args = parser.parse_args()

    # experiment name
    if args.exp_name:
        exp_folder = args.exp_name
    else:
        exp_folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(
        f"[INFO] Starting {args.runs} CMA-ES runs for experiment '{exp_folder}'...")
    for run_number in range(1, args.runs + 1):
        run_output_folder = os.path.join(exp_folder, f"run_{run_number}")
        print(
            f"\n--- Starting Run {run_number}/{args.runs} (Output to: data/genomes/{run_output_folder}) ---")

        run_cmaes_difftaichi.run(
            mode=args.mode,
            gens=args.gens,
            sigma_val=args.sigma,
            hidden_sizes=args.hidden_sizes,
            output_folder=run_output_folder
        )

    # visualization
    if args.visualize_best:
        print("[INFO] Launching live visualization of best individual...")
        print(
            "[WARN] Visualization may need to be adapted to read from the new directory structure.")

        logs_bool = args.logs.lower() in ('yes', 'true', 't', '1')

        process = multiprocessing.Process(
            target=best_individual_latest_difftaichi.visualize_best,
            args=(args.mode, logs_bool))
        process.start()
        time.sleep(2)

    # plotting
    if args.plot:
        print("[INFO] Plotting fitness over generations for the first run...")
        try:
            filename = os.path.join(
                "data", "genomes", exp_folder, "run_1", "output.csv")
            plot_fitness_difftaichi.plot(filename)
        except Exception as e:
            print(f"[WARN] Could not plot fitness from '{filename}': {e}")

    finished_signal_path = os.path.join(
        "data", "genomes", exp_folder, "_FINISHED")
    try:
        with open(finished_signal_path, 'w') as f:
            pass
        print(f"[INFO] Created finish signal file at: {finished_signal_path}")
    except Exception as e:
        print(f"[WARN] Could not create finish signal file: {e}")

    print("\n[INFO] Experiment completed.")

    # if visualization is running, wait until its done
    if 'process' in locals() and process.is_alive():
        print("[INFO] Waiting for visualization process to terminate...")
        process.join(timeout=10)
        if process.is_alive():
            print(
                "[WARN] Visualization process did not terminate gracefully, forcing it to close.")
            process.terminate()


if __name__ == "__main__":
    main()
