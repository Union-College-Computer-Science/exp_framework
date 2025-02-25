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

import csv
import os
import argparse
from datetime import datetime
from pathlib import Path
from cmaes import CMA
import numpy as np
from snn_sim.run_simulation import run

SNN_INPUT_SHAPE = 36
MEAN_ARRAY = [1.0] * SNN_INPUT_SHAPE
NUM_ITERS = 500

VERBOSE = False

GENOME_INDEX = 0
FITNESS_INDEX = 1

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATE_TIME = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

def objective(x):
    """
    Used to evaluate the fitness of a genome.

    Parameters:
        x (ndarray): Genome.

    Returns:
        float: Fitness score to minimize.
    """
    if VERBOSE:
        print("Evaluating genome:", x)
    try:
        fitness, total_fires = run(NUM_ITERS, x, "h")
        if VERBOSE:
            print(f"Fitness: {fitness}, Total SNN Fires: {total_fires}")
        return fitness
    except Exception as e:
        print("Error running simulation:", e)
        return float('inf')  # Return a high value to indicate failure

def run_cma_es(mode, gens, sigma_val):
    """
    Runs the cma_es algorithm on the robot locomotion problem,
    with sin-like robot actuators. Saves a csv file to ./output
    with each robot's genome & fitness for every generation.

    Parameters:
        mode (string): How to run the simulation. 
                       "headless" runs without any video or visual output.
                       "video" outputs the simulation as a video in the "./videos folder.
                       "screen" shows the simulation on screen as a window.
                       "both: shows the simulation on a window and saves a video.
        gens (int): How many generations to run.
        sigma_val (float): The standard deviation of the normal distribution
        used to generate new candidate solutions
    """

    # Generate output.csv file
    csv_header = ['generation', 'best_fitness']
    csv_header.extend([f"weight{i}" for i in range(SNN_INPUT_SHAPE)])

    Path(os.path.join(ROOT_DIR, "data")).mkdir(parents=True, exist_ok=True)

    csv_name = DATE_TIME + ".csv"
    csv_path = os.path.join(ROOT_DIR, "data", csv_name)

    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)

    # Init CMA
    optimizer = CMA(mean=np.array(MEAN_ARRAY), sigma=sigma_val)

    for generation in range(gens):
        solutions = []

        for indv_num in range(optimizer.population_size):
            x = optimizer.ask()
            fitness = objective(x)
            solutions.append((x, fitness))

        best_fitness = solutions[0][FITNESS_INDEX]
        best_genome = solutions[0][GENOME_INDEX]

        optimizer.tell(solutions)

        if VERBOSE:
            print([i[1] for i in solutions])

        print("Generation", generation, "Best Fitness:", best_fitness)

        # Add a new row to output.csv file with cols: generation#, fitness, and genome
        new_row = [generation, best_fitness] + best_genome.tolist()

        with open(csv_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(new_row)

        # If --mode s, v, or b show/save best individual from generation
        if mode in ["s", "b", "v"]:
            vid_name = DATE_TIME + "_gen" + str(generation)
            vid_path = os.path.join(ROOT_DIR, "videos", DATE_TIME)

            run(NUM_ITERS, best_genome, mode, vid_name, vid_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--mode',  #headless, screen, video, both h, s, v, b
        help='mode for output. h-headless , s-screen, v-video, b-both',
        default="h")
    parser.add_argument('--gens',
                        type=int,
                        help='number of generations to run',
                        default=100)
    parser.add_argument('--sigma',
                        type=int,
                        default=2,
                        help='sigma value for cma-es')
    args = parser.parse_args()

    run_cma_es(args.mode, args.gens, args.sigma)
