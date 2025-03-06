"""
Visualize the best individual so far, pulling from output.csv.

Author: James Gaskell
February 6th, 2025
"""

import os
import argparse
from pathlib import Path
import glob
import pandas
from snn_sim.run_simulation import run

ITERS = 1000
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

PARENTDIR = Path(__file__).parent.resolve()
GENOME_START_INDEX = 3
FILEFOLDER = "data"

def visualize_best(filename, mode, gen):
    """
    Look at output.csv and continuously run best individual.
    Assumes csv names are their best achieved fitnesses
    Continually searches for the lowest best fitness, plays the visualization and repeats
    """

    while True:
        path = os.path.join(ROOT_DIR, "data", filename)
        df = pandas.read_csv(path)
        if gen != None:
            row = df.loc[(df['generation']==gen)]
            genome = row.values.tolist()[0][GENOME_START_INDEX:]
        else:
            best_fitness = min(df["best_fitness"])
            row = df.loc[df['best_fitness'] == best_fitness]
            genome = row.values.tolist()[0][GENOME_START_INDEX:]
        if mode == "s":
            run(ITERS, genome, "s")
        elif mode == "v":
            run(ITERS, genome, "v", "cur_example", "videos")
            print("SAVED VIDEO")
            quit()

def get_latest():
    """
    Finds the most recently created file in a folder matching a given pattern.
    
    Returns:
        str: The path to the most recently created file, or None if no files match the pattern.
    """
    this_dir = Path(__file__).parent.resolve()
    files = glob.glob(os.path.join(this_dir, FILEFOLDER) + "/*.csv")
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime).split("/")[-1]
    return latest_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--mode',  #headless, screen, video, both h, s, v, b
        help='mode for output. s-screen, v-video',
        default="s")
    parser.add_argument('--gen',
                        type=int,
                        help='generation number to visualize',
                        default=None)
    args = parser.parse_args()

    visualize_best(get_latest(), args.mode, args.gen)
