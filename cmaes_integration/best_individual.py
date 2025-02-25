"""
Simple script to run the best individual from an output CSV file.

Author: Thomas Breimer
January 29th, 2025
"""

import os
import argparse
import pathlib
import pandas as pd
from snn_sim.run_simulation import run

ITERS = 1000

def run_best(filename):
    """
    Run the best individual from a csv file.
    
    Parameters:
        filename (string): CSV file to look at. Should be in /data directory.
    """
    
    this_dir = pathlib.Path(__file__).parent.resolve()
    df = pd.read_csv(os.path.join(this_dir, os.path.join("data", filename)))
    
    # Get the individual with the highest (or lowest) fitness
    # Assuming lower fitness is better (minimization problem)
    best_row = df.loc[df['fitness'].idxmin()]
    best_gen = best_row['generation']
    best_fitness = best_row['fitness']
    
    # Get the genome
    genome = best_row.values.tolist()[2:]  # Skip generation and fitness columns
    
    print(f"\n===== Running Best Individual =====")
    print(f"From Generation: {best_gen}")
    print(f"Fitness: {best_fitness}")
    print(f"The simulation will display SNN firing statistics after completion")
    print(f"This will show you how often each SNN neuron fired during the run\n")
    
    # Run the simulation
    fitness, total_fires = run(ITERS, genome, "s")
    
    print(f"\n===== Results Summary =====")
    print(f"Generation: {best_gen}")
    print(f"Original Fitness: {best_fitness}")
    print(f"Current Fitness: {fitness}")
    print(f"Total SNN Fires: {total_fires}")
    
    return fitness, total_fires, best_gen

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the best individual from a CSV file')
    
    parser.add_argument(
        '--filename',
        type=str,
        help='CSV file to analyze (must be in data directory)',
        required=True)
    
    args = parser.parse_args()
    
    run_best(args.filename)
