"""
Visualize the best individual so far during a DiffTaichi run.
It finds the latest experiment folder and polls the CSV files within it
to display the best-performing genome in real-time.

Author: James Gaskell, Thomas Breimer
Modified for DiffTaichi by Takumi Kojima
June 4th, 2025
"""

import os
import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

import diffnpm

ROOT_DIR = Path(__file__).parent.resolve()
DATA_ROOT = ROOT_DIR / "data" / "genomes"
GENOME_START_INDEX = 2


def get_latest_experiment_folder():
    """get the latest experimtn folder in data/genomes/"""
    try:
        all_exp_dirs = [d for d in DATA_ROOT.iterdir() if d.is_dir()]
        if not all_exp_dirs:
            return None
        latest_dir = max(all_exp_dirs, key=os.path.getctime)
        return latest_dir
    except FileNotFoundError:
        return None


def get_best_from_all_csvs(experiment_folder):
    """
    search for the all runs in the given experiment folder,
    and return the row (genome) which has the best fitness
    """
    best_row = None
    best_fitness = float('inf')
    best_file = None

    if not experiment_folder:
        return None, None

    # search run_* folder
    run_dirs = [d for d in experiment_folder.iterdir() if d.is_dir()
                and d.name.startswith('run_')]

    for run_dir in run_dirs:
        csv_file = run_dir / "output.csv"
        if not csv_file.exists():
            continue

        try:
            df = pd.read_csv(csv_file)
            if df.empty or "best_fitness" not in df.columns:
                continue

            # get the best fitness
            current_best_row = df.loc[df["best_fitness"].idxmin()]
            current_fitness = current_best_row["best_fitness"]

            # compare the best among all and update
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_row = current_best_row
                best_file = csv_file
        except Exception as e:
            print(f"Skipping file {csv_file} due to error: {e}")

    return best_row, best_file


def visualize_best():
    """
    watch the experiment folder and visulize the best fitness in live.
    """
    print("Initializing DiffTaichi for visualization...")
    diffnpm.initialize()

    last_best_genome = None
    final_visualization_done = False

    while True:
        latest_exp_folder = get_latest_experiment_folder()
        if not latest_exp_folder:
            print(f"Waiting for experiment data in {DATA_ROOT}...")
            time.sleep(5)
            continue

        is_experiment_finished = False
        finish_signal_file = latest_exp_folder / "_FINISHED"
        if finish_signal_file.exists():
            is_experiment_finished = True
            if not final_visualization_done:
                print(
                    "\n[INFO] Experiment finish. Performing final visualization...")

        best_row, source_file = get_best_from_all_csvs(latest_exp_folder)

        if best_row is not None:
            genome = best_row.iloc[GENOME_START_INDEX:].values.astype(
                np.float32)

            if last_best_genome is None or not np.array_equal(genome, last_best_genome):
                last_best_genome = genome
                print("\n" + "="*50)
                print(f"New best Found. Visualizing...")
                print("="*50 + "\n")

                try:
                    diffnpm.live_visualize(genome)
                except Exception as e:
                    print(f"Error during visualization: {e}")
            else:
                if not is_experiment_finished:
                    print(
                        f"\rBest from {latest_exp_folder.name} is unchanged. Waiting...", end="")
        else:
            if not is_experiment_finished:
                print(
                    f"\rWaiting for valid CSV in {latest_exp_folder}...", end="")

        if is_experiment_finished:
            if not final_visualization_done:
                if last_best_genome is not None:
                    print(
                        "[INFO] Running final visualization.")
                    diffnpm.live_visualize(last_best_genome)
                final_visualization_done = True

            print("[INFO] Monitoring finished. Exiting.")
            break

        time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize the best individual from a run.')
    parser.add_argument(
        '--mode',
        choices=["s", "h"],
        help='mode for output. s-screen',
        default="s")
    parser.add_argument(
        '--logs', type=str, help='(Not implemented for DiffTaichi)', default="False")

    args = parser.parse_args()

    if args.mode == 'h':
        print("Running in headless mode. No visualization will be shown.")
    else:
        visualize_best(args.mode, False)
