"""
Plot fitness over generations from a Difftaichi experiment folder.
It reads output.csv from all 'run_*' subdirectories, and plots the
mean fitness with a min-max range.

Takes a command line argument of the experiment folder name.

Author: Thomas Breimer
February 10th, 2025

Modified for DiffTaichi by Takumi Kojima
June 6th, 2025
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot(exp_folder):
    """
    Plots fitness over generations from an experiment folder.
    results from multiple runs if available.

    Args:
        exp_folder (str): Name of the experiment folder inside 'data/genomes/'.
    """

    exp_path = Path("data") / "genomes" / exp_folder

    if not exp_path.is_dir():
        print(f"Error: Experiment folder not found at '{exp_path}'")
        return

    # Find all output.csv files within run_* subdirectories
    csv_files = list(exp_path.glob("run_*/output.csv"))

    if not csv_files:
        print(
            f"Error: No 'output.csv' files found inside {exp_path}/run_*.")
        return

    all_runs_fitness = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                print(f"Warning: Skipping empty file {csv_file}")
                continue

            numeric_fitness = pd.to_numeric(
                df['best_fitness'], errors='coerce')
            best_so_far = numeric_fitness.dropna().cummin()

            all_runs_fitness.append(best_so_far)
        except Exception as e:
            print(f"Warning: Could not read or process {csv_file}. Error: {e}")

    if not all_runs_fitness:
        print("Error: No valid fitness data could be processed.")
        return

    fitness_df = pd.concat(all_runs_fitness, axis=1)
    num_runs = len(fitness_df.columns)

    generations = fitness_df.index

    # Calculate statistics across all runs
    mean_fitness = fitness_df.mean(axis=1)
    min_fitness = fitness_df.min(axis=1)
    max_fitness = fitness_df.max(axis=1)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot the mean fitness line
    plt.plot(generations, mean_fitness, marker='o', linestyle='-',
             color='b', label=f"Mean Fitness ({num_runs} runs)")

    # Shade the area between min and max fitness
    plt.fill_between(generations, min_fitness, max_fitness,
                     color='b', alpha=0.2, label="Min-Max Range")

    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title(f"Fitness Evolution for Experiment: '{exp_folder}'")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot fitness from a DiffTaichi experiment.')

    parser.add_argument(
        '--exp_folder',
        help='Name of the experiment folder in data/genomes/',
        required=True)

    args = parser.parse_args()

    plot(args.exp_folder)
