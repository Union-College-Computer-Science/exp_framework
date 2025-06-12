
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from cmaes import SepCMA

import sys
import os

import diffnpm

POP_SIZE = 5
ITERS = 1000
NUM_ACTUATORS = 4
INPUT_SIZE = 2
OUTPUT_SIZE = 1
HIDDEN_SIZES = [10]
GENOME_LENGTH = None

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATE_TIME = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
FITNESS_INDEX = 1
GENOME_INDEX = 0


def run(mode, gens, sigma_val, hidden_sizes, output_folder=DATE_TIME):
    global GENOME_LENGTH
    layer_input_size = INPUT_SIZE
    params_per_snn = 0

    for hidden_size in hidden_sizes:
        params_per_snn += (layer_input_size + 1) * hidden_size
        layer_input_size = hidden_size

    params_per_snn += (layer_input_size + 1) * OUTPUT_SIZE
    GENOME_LENGTH = NUM_ACTUATORS * params_per_snn

    MEAN_ARRAY = [0.0] * GENOME_LENGTH
    bounds = [(-100.0, 100.0)] * GENOME_LENGTH

    output_dir = os.path.join(ROOT_DIR, "data", "genomes", output_folder)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "output.csv")

    csv_header = ["generation", "best_fitness"]
    csv_header.extend([f"weight_{i}" for i in range(GENOME_LENGTH)])
    pd.DataFrame(columns=csv_header).to_csv(csv_path, index=False)

    optimizer = SepCMA(mean=np.array(MEAN_ARRAY), sigma=sigma_val,
                       bounds=np.array(bounds), population_size=POP_SIZE)

    best_fitness_so_far = float("inf")

    diffnpm.initialize()

    for generation in range(gens):
        solutions = []
        for _ in range(optimizer.population_size):
            genome = optimizer.ask()
            fitness = diffnpm.run(genome)
            solutions.append((genome, fitness))

        optimizer.tell(solutions)
        sorted_solutions = sorted(solutions, key=lambda x: x[FITNESS_INDEX])
        best_sol = sorted_solutions[0]

        # store the best genome in each 10 gens
        if generation % 10 == 0 or generation == gens - 1:
            print(f"Visualizing best genome of generation {generation}...")
            # create the folder to store
            vis_folder = os.path.join(
                output_dir, f"generation_{generation:04d}")
            best_genome = best_sol[GENOME_INDEX]
            # call visualize func
            diffnpm.run_and_visualize(best_genome, vis_folder)

        if best_sol[FITNESS_INDEX] < best_fitness_so_far:
            best_fitness_so_far = best_sol[FITNESS_INDEX]

        print(
            f"Generation {generation}: Best Fitness = {best_sol[FITNESS_INDEX]}")

        row = [generation, best_sol[FITNESS_INDEX]] + \
            best_sol[GENOME_INDEX].tolist()
        pd.DataFrame([row]).to_csv(
            csv_path, mode='a', header=False, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gens', type=int, default=50)
    parser.add_argument('--sigma', type=float, default=1.0)
    args = parser.parse_args()

    run(mode="h", gens=args.gens, sigma_val=args.sigma, hidden_sizes=HIDDEN_SIZES)
