"""
Runs cma-es on `run_simulation.py` as a fitness function.
Creates output.csv and updates it continuously with the best individual from each generation.
Whether to show the simulation or save as video, number of generations, sigma can be passed as
command line arguments. Example: `python3 run_cmaes.py --gens 50 --sigma 2 --mode h` 
runs cma-es for 50 generations
in headless mode with a sigma of 2. Replacing "--mode h" with "--mode s" makes the simulation 
output to the screen, and replacing it with "--mode v" saves each simulation 
as a video in `./videos`. 
"--mode b" shows on screen and saves a video.

Authors: Thomas Breimer, James Gaskell
February 4th, 2025
"""

import os
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
from cmaes import SepCMA
import numpy as np
from snn_sim import run_simulation
import os
import sys

def is_windows():
    """
    Checks if the operating system is Windows.

    Returns:
        bool: True if the OS is Windows, False otherwise.
    """
    return os.name == 'nt' or sys.platform.startswith('win')

NUM_ACTUATORS = 8
POP_SIZE = 12

INPUT_SIZE = 2
OUTPUT_SIZE = 1

# Shape of the genome
SNN_INPUT_SHAPE = 72

# Mean genome
MEAN_ARRAY = [0.0] * SNN_INPUT_SHAPE

# Num of sim time steps
ITERS = 1000

VERBOSE = False

# Index of genome & fitness in tuple
GENOME_INDEX = 0
FITNESS_INDEX = 1

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATE_TIME = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def run(mode, gens, sigma_val, hidden_sizes, output_folder=DATE_TIME, run_number=1, spike_decay=PIKE_DECAY_DEFAULT, robot_config_path=None):
    """
    Runs the cma_es algorithm on the robot locomotion problem,
    with sin-like robot actuators. Saves a csv file to ./output
    with each robot's genome & fitness for every generation.

    Parameters:
        mode (string): How to run the simulation. 
                       "h" runs without any video or visual output.
                       "v" outputs the simulation as a video in the "./videos folder.
                       "s" shows the simulation on screen as a window.
                       "b: shows the simulation on a window and saves a video.
        gens (int): How many generations to run.
        sigma_val (float): The standard deviation of the normal distribution
        used to generate new candidate solutions
    """

    # Generate output.csv file
    csv_header = ['generation', 'best_fitness', "best_so_far"]
    csv_header.extend([f"weight{i}" for i in range(SNN_INPUT_SHAPE)])

    Path(os.path.join(ROOT_DIR, "data", "genomes")).mkdir(parents=True, exist_ok=True)

    if output_folder is None:
        output_folder = DATE_TIME

    csv_path = os.path.join(ROOT_DIR, "data", "genomes", output_folder)
    Path(csv_path).mkdir(parents=True, exist_ok=True)
    csv_filename = f"run_{run_number}.csv"

    # Set up symlink to output folder (once, only for run_1)
    symlink_path = os.path.join(ROOT_DIR, "data","latest_genome")

    if run_number == 1:
        try:
            if os.path.islink(symlink_path) or os.path.exists(symlink_path):
                os.remove(symlink_path)

            if is_windows():
                os.symlink(csv_path, symlink_path)
            else:
                os.system(f'ln -s "{csv_path}" "{symlink_path}"')

        except Exception as e:
            print(f"Warning: could not create symlink to latest_genome folder: {e}")

    pd.DataFrame(columns=csv_header).to_csv(os.path.join(csv_path, csv_filename), index=False)

    # Perhaps try bounds again? doesn't seem to be doing anything
    # YES! This works
    bounds = [(-100000, 100000)] * SNN_INPUT_SHAPE
    for i in range(len(bounds)):
       if (i+1) % 3 == 0:
           bounds[i] = (0, 200000)

    # Init CMA
    optimizer = SepCMA(mean=np.array(MEAN_ARRAY), sigma=sigma_val, bounds=np.array(bounds), population_size=POP_SIZE)

    best_fitness_so_far = run_simulation.FITNESS_OFFSET

    # Run generations
    for generation in range(gens):
        solutions = []

        # Run individuals
        for _ in range(optimizer.population_size):
            x = optimizer.ask() # Ask cmaes for a genome
            fitness = run_simulation.run(ITERS, x, "h", hidden_sizes) # get fitness
            solutions.append((x, fitness))

        optimizer.tell(solutions) # Tell cmaes about population

        sorted_solutions = sorted(solutions, key=lambda x: x[FITNESS_INDEX])

        best_sol = sorted_solutions[0]

        if best_sol[FITNESS_INDEX] < best_fitness_so_far:
            print("Found new best! Old:", best_fitness_so_far, "New:", best_sol[FITNESS_INDEX])
            best_fitness_so_far = best_sol[FITNESS_INDEX]

        if VERBOSE:
            print([i[1] for i in sorted_solutions])

        print("Generation", generation, "Best Fitness:", best_sol[FITNESS_INDEX])

        # Add a new row to output.csv file with cols: generation#, fitness, and genome
        new_row = [generation, best_sol[FITNESS_INDEX], best_fitness_so_far] + \
            best_sol[GENOME_INDEX].tolist()

        new_row_df = pd.DataFrame([new_row], columns=csv_header)

        # Append the new row to the CSV file using pandas in append mode (no header this time).
        new_row_df.to_csv(os.path.join(csv_path, csv_filename), mode='a', index=False, header=False)

        # If --mode s, v, or b show/save best individual from generation
        if mode in ["s", "b", "v"]:
            vid_name = DATE_TIME + "_gen" + str(generation)
            vid_path = os.path.join(ROOT_DIR, "data", "videos", DATE_TIME)

<<<<<<< HEAD
            run_simulation.run(ITERS, best_sol[GENOME_INDEX], mode, hidden_sizes, vid_name, vid_path)
=======
            run_simulation.run(
                ITERS, best_sol[GENOME_INDEX], mode, vid_name, vid_path)
>>>>>>> parent of b03e1aa (Merge branch 'main' into run_experiment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--mode',  # headless, screen, video, both h, s, v, b
        help='mode for output. h-headless , s-screen, v-video, b-both',
        default="h")
    parser.add_argument('--gens',
                        type=int,
                        help='number of generations to run',
                        default=500)
    parser.add_argument('--sigma',
                        type=float,
                        default=100,
                        help='sigma value for cma-es')
    args = parser.parse_args()

    run(args.mode, args.gens, args.sigma)
