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
import time
import argparse
import multiprocessing
from datetime import datetime
from pathlib import Path
from collections import Counter
import pandas as pd
from cmaes import CMA
import numpy as np
from snn_sim import run_simulation


SNN_INPUT_SHAPE = 72
MEAN_ARRAY = [0.0] * SNN_INPUT_SHAPE
NUM_ITERS = 1000

VERBOSE = False

GENOME_INDEX = 0
FITNESS_INDEX = 1

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATE_TIME = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

def run(mode, gens, sigma_val):
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
    csv_header = ['generation', 'best_fitness', "best_so_far"]
    csv_header.extend([f"weight{i}" for i in range(SNN_INPUT_SHAPE)])

    Path(os.path.join(ROOT_DIR, "data")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(ROOT_DIR, "action_log", f"{DATE_TIME}")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(ROOT_DIR, "levels_log", f"{DATE_TIME}")).mkdir(parents=True, exist_ok=True)

    csv_path = os.path.join(ROOT_DIR, "data", f"{DATE_TIME}.csv")

    #if os.path.exists("latest.csv"):
    #    os.remove("latest.csv")

    # os.system("ln -s " + csv_path + " latest.csv")

    pd.DataFrame(columns=csv_header).to_csv(csv_path, index=False)

    # Init CMA
    optimizer = CMA(mean=np.array(MEAN_ARRAY), sigma=sigma_val, population_size=25)

    best_fitness_so_far = run_simulation.FITNESS_OFFSET

    for generation in range(gens):
        Path(os.path.join(ROOT_DIR, "levels_log", f"{DATE_TIME}",
                          f"gen_{generation}")).mkdir(parents=True, exist_ok=True)
        solutions = []

        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            fitness, action_log, levels_log = run_simulation.run(NUM_ITERS, x, "h")
            solutions.append((x, fitness))

        action_log = np.array(action_log).T
        action_log_csv = pd.DataFrame()
        for i, x in enumerate(action_log):
            z = dict(Counter(list(map(lambda y: round(y, 2), x))))
            # print(f"Firing freq SNN {i}: {z}")
            temp = pd.DataFrame(z.values(), index=z.keys(), columns=[i])
            action_log_csv = pd.concat([action_log_csv, temp], axis=1).sort_index()
        action_log_csv.to_csv(os.path.join(ROOT_DIR, "action_log", f"{DATE_TIME}",
                                    f"gen_{generation}.csv"), index=True)

        for i, x in levels_log.items():
            levels_log_csv = pd.DataFrame()
            for name, layer in x.items():
                for node in layer:
                    temp = pd.DataFrame(node, columns=[name])
                    levels_log_csv = pd.concat([levels_log_csv, temp], axis=1).sort_index()
                    levels_log_csv.to_csv(os.path.join(ROOT_DIR, "levels_log", f"{DATE_TIME}",
                                            f"gen_{generation}", f"snn_{i}.csv"), index=True)

        optimizer.tell(solutions)

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
        new_row_df.to_csv(csv_path, mode='a', index=False, header=False)

        if generation > 50 and best_fitness_so_far > 97:
            return # Prune bad experiments

        # If --mode s, v, or b show/save best individual from generation
        if mode in ["s", "b", "v"]:
            vid_name = DATE_TIME + "_gen" + str(generation)
            vid_path = os.path.join(ROOT_DIR, "videos", DATE_TIME)

            run_simulation.run(NUM_ITERS, best_sol[GENOME_INDEX], mode, vid_name, vid_path)

def task(name, mode, gens, sigma): # Wait until all processes are ready to start
    print(f"Thread {name}: starting")
    run(mode, gens, sigma)
    print(f"Thread {name}: finishing")
    return name  # Return the thread name so it can be restarted

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
                        default=3,
                        help='sigma value for cma-es')
    args = parser.parse_args()

    # run(args.mode, args.gens, args.sigma)

    num_workers = multiprocessing.cpu_count()

    while True:
        with multiprocessing.Pool(processes=num_workers) as pool:
            # Apply async tasks with the start_event
            for name in ["A", "B", "C", "D", "E", "F", "G", "H"]:
                pool.apply_async(task, args=(name, args.mode, args.gens, args.sigma))

            pool.close()  # Prevents new tasks from being manually submitted
            
            pool.join()  # Wait for all processes to finish