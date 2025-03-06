"""
Runs the latest experiment. Also launches best_individual.py to show the best individual so far
and plots fitness over generations after the run is over.

Whether to show the simulation or save as video, number of generations, sigma can be passed as
command line arguments. Example: `python3 run_cmaes.py --gens 50 --sigma 2 --mode h` 
runs cma-es for 50 generations
in headless mode with a sigma of 2. Replacing "--mode h" with "--mode s" makes the simulation 
output to the screen, and replacing it with "--mode v" saves each simulation 
as a video in `./videos`.  "--mode b" shows on screen and saves a video.

By Thomas Breimer
March 6th, 2025
"""

import argparse
import threading
import run_cmaes
import best_individual
import plot_fitness_over_gens

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
                        type=float,
                        default=1,
                        help='sigma value for cma-es')
    args = parser.parse_args()

    t1 = threading.Thread(target=best_individual.visualize_best)

    t1.start()
    run_cmaes.run(args.mode, args.gens, args.sigma)

    t1.join()
    plot_fitness_over_gens.plot()
