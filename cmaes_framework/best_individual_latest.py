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
import numpy as np
from matplotlib import pyplot as plt
from snn_sim.run_simulation import run

ITERS = 1000
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

PARENTDIR = Path(__file__).parent.resolve()
GENOME_START_INDEX = 3
FILEFOLDER = "data"



def visualize_best(graphs, mode, filename="latest.csv"):
    """
    Look at a csv and continuously run best individual.
    
    Parameters:
        filename (str): Filename of csv to look at. Defaults to latest.csv.
    """

    #time.sleep(10)

    path = os.path.join(PARENTDIR, filename)

    while True:
        if os.path.exists(path):
            df = pandas.read_csv(path)

            best_fitness = min(df["best_fitness"])
            row = df.loc[df['best_fitness'] == best_fitness]
            genome = row.values.tolist()[0][GENOME_START_INDEX:]
            generation = row.values.tolist()[0][0]

            # Make video directory if we're making a video.
            if mode in ["v", "b"]:
                os.makedirs("videos", exist_ok=True)
                this_dir = pathlib.Path(__file__).parent.resolve()
                vid_name = filename + "_gen" + str(generation)
                vid_path = os.path.join(this_dir, "videos")
                _, spikes, levels = run(ITERS, genome, "v", vid_name, vid_path)
                quit()
            else:
                _, spikes, levels = run(ITERS, genome, "s")



            spikes = np.array(spikes)
            levels = np.array([[x[0] for x in row] for row in levels])

        if graphs == "s":

            fig, ax = plt.subplots(figsize=(12, 5))

            for neuron_idx in range(spikes.shape[1]):
                spike_times = np.where(spikes[:, neuron_idx])[0]
                ax.vlines(spike_times, neuron_idx - 0.4, neuron_idx + 0.4)

            ax.set_yticks(np.arange(spikes.shape[1]))
            ax.set_yticklabels([f'Neuron {i}' for i in range(spikes.shape[1])])
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Neuron Index")
            ax.set_title("Spike Train of Neurons")
            plt.tight_layout()
            plt.xlim(0, 100)
            plt.show(block=True)
        
        elif graphs == "l":

            fig, ax = plt.subplots(figsize=(12, 6))
            lines = []

            # Plot and make lines pickable
            for i in range(levels.shape[1]):
                line, = ax.plot(levels[:, i], label=f'Neuron {i}', picker=True, pickradius=5, alpha=0.4)
                lines.append(line)

            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Level')
            ax.set_title('Click a Line to Highlight It')
            ax.legend(loc='upper right')
            plt.tight_layout()

            def on_pick(event):
                # Reset all lines
                for line in lines:
                    line.set_linewidth(1.5)
                    line.set_alpha(0.4)

                # Highlight selected line
                picked_line = event.artist
                picked_line.set_linewidth(3)
                picked_line.set_alpha(1.0)
                fig.canvas.draw()

            fig.canvas.mpl_connect('pick_event', on_pick)

            plt.show(block=True)

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--graphs',
        help='graph outputs and levels? n - no, s - spike trains, l - levels, b - both',
        default="n")
    parser.add_argument(
        '--mode', #headless, screen, video, both h, s, v, b
        help='mode for output. h-headless , s-screen, v-video, b-both',
        default="s")

    
    args = parser.parse_args()
    
    visualize_best(args.graphs, args.mode)

        
    