"""
Visualize the best individual so far, pulling from output.csv.

Author: James Gaskell
February 6th, 2025
"""

import os
import time
import argparse
import pandas
from rsnn_sim.run_sim_rsnn import run

ITERS = 500
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def visualize_best(filename="latest.csv"):
    """
    Look at latest.csv and continuously run best individual.
    Assumes csv names are their best achieved fitnesses
    Continually searches for the lowest best fitness, plays the visualization and repeats
    """

    if filename == "latest.csv":
        path = "latest.csv"
    else:
        path = os.path.join(ROOT_DIR, "data", filename)

    time.sleep(1)

    while True:
        if os.path.exists(path):
            df = pandas.read_csv(path)

            if len(df["best_fitness"] > 0):
                best_fitness = min(df["best_fitness"])
                row = df.loc[df['best_fitness'] == best_fitness]
                genome = row.values.tolist()[0][3:]
                run(ITERS, genome, "s")
        else:
            time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--file',
                        type=str,
                        default=None,
                        help='csv file to run')

    args = parser.parse_args()

    if args.file is None:
        raise Exception('No csv file specified!')

    visualize_best(args.file)
