"""
Run a single individual from its genome in an output csv file.
Takes one command line arg "--gen" corresponding to generation number.
Takes another command line arg "--mode" which displays the simulation in different ways.
"--mode s" makes the simulation output to the screen, replacing it with "--mode v" saves 
each simulation as a video in `./videos`. "-mode b" shows on screen and saves a video.
Must also specify --filename for csv file.

Run with command line argument `--logs true` to create SNN logs.
A CSV file will be generated in /cmaes_framework/snn_logs.

Author: Thomas Breimer
January 29th, 2025
"""

import os
import argparse
import pathlib
import pandas as pd
from snn_sim.run_simulation import run

ITERS = 100
GENOME_START_INDEX = 3

def run_indvididual(generation, mode, filename, logs):
    """
    Run an individual from a csv file.
    
    Parameters:
        generation (int): Generation number of individual.
        mode (string): Tells whether to show simulation, save it to
                       video, or both. "screen" renders the video to the screen. "video" saves a
                       video to the "./videos" folder. "both" does both of these things.
        filename (string): CSV file to look at. Should be in cmaes_framework/data directory.
        logs (bool): Whether or not to produce SNN logs.
    """

    # Make video directory if we're making a video.
    if mode in ["v", "b"]:
        os.makedirs(os.path.join("data", "videos"), exist_ok=True)

    this_dir = pathlib.Path(__file__).parent.resolve()

    # Read genome from csv file
    if filename == "latest.csv":
        csv_path = os.path.join(this_dir, "latest.csv")
    else:
        csv_path = os.path.join(this_dir, os.path.join("data", filename))
    
    df = pd.read_csv(csv_path)
    row = df.loc[(df['generation']==generation)]
    genome = row.values.tolist()[0][GENOME_START_INDEX:]

    # Generate video name using times
    vid_name = filename + "_gen" + str(generation)
    vid_path = os.path.join(this_dir, "data", "videos")

    run(ITERS, genome, mode, vid_name, vid_path, logs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument(
        '--mode', #headless, screen, video, both h, s, v, b
        help='mode for output. h-headless , s-screen, v-video, b-both',
        default="s")

    parser.add_argument(
        '--gen',
        type=int,
        help='what generation to grab',
        default=1)

    parser.add_argument(
        '--filename',
        type=str,
        help='what csv file to look at',
        default="latest.csv")
    
    parser.add_argument(
        '--logs',
        type=str,
        help='whether to generate SNN logs (true/false)',
        default="false")

    args = parser.parse_args()

    run_indvididual(args.gen, args.mode, args.filename, bool(args.logs))
