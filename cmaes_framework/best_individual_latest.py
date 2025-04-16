"""
Visualize the best individual so far during a run, pulling from latest.csv.

Author: James Gaskell, Thomas Breimer
April 3rd, 2025
"""

import os
from pathlib import Path
import pandas
import argparse
import pathlib
import time
import numpy as np
from matplotlib import pyplot as plt
from snn_sim.run_simulation import run

ITERS = 1000
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

PARENTDIR = Path(__file__).parent.resolve()
GENOME_START_INDEX = 3
FILEFOLDER = "data"

def wait_for_file(path):
    """
    Wait for a file to be created and contain valid 'best_fitness' data.

    Parameters:
        get_df_func (function): A function that returns the DataFrame to check.
    """
    while True:
        try:
            df = pandas.read_csv(path)
            if not df.empty and "best_fitness" in df.columns:
                _ = min(df["best_fitness"])
                return df
        except ValueError as e:
            pass

        print("Waiting...")
        time.sleep(5)



def visualize_best(mode, logs, filename="latest.csv"):
    """
    Look at a csv and continuously run best individual.
    
    Parameters:
        filename (str): Filename of csv to look at. Defaults to latest.csv.
    """

    path = os.path.join(PARENTDIR, filename)
    os.makedirs("data", exist_ok=True)

    while True:
        if os.path.exists(path):
            df = wait_for_file(path)

            try:
                best_fitness = min(df["best_fitness"])
                row = df.loc[df['best_fitness'] == best_fitness]
                genome = row.values.tolist()[0][GENOME_START_INDEX:]
                generation = int(row.values.tolist()[0][0])
                this_dir = pathlib.Path(__file__).parent.resolve()
                vid_path = os.path.join(this_dir, "data", "videos")

                print("\n\n\nFitness: ", best_fitness)

                real_filename = Path(path).resolve().name.split(".")[0]
                vid_name = real_filename + "_gen_" + str(generation)

                # Make video directory if we're making a video.
                if mode in ["v", "b"]:
                    os.makedirs(os.path.join("data", "videos"), exist_ok=True)
                    run(ITERS, genome, mode, vid_name, vid_path, logs, (real_filename + ".csv"))
                    quit()
                elif mode in ["s","h"]:
                    run(ITERS, genome, mode, None, None, logs, (real_filename + ".csv"))
                    if logs:
                        quit()

            except ValueError as e:
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--mode', #headless, screen, video, both h, s, v, b
        help='mode for output. h-headless , s-screen, v-video, b-both',
        default="s")
    parser.add_argument(
        '--logs',
        type=str,
        help='whether to generate SNN logs (true/false)',
        default="True")

    
    args = parser.parse_args()

    logs = args.logs

    if logs.lower() in ('yes', 'true', 't', '1'):
        logs = True
    elif logs.lower() in ('no', 'false', 'f', '0'):
        logs = False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

    
    visualize_best(args.mode, logs)

        
    