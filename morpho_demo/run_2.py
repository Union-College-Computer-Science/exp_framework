"""
Simple RMHC of walking robot from scratch in evogym

Author: Thomas Breimer, Matthew Meek
January 22nd, 2025
"""

import os
import cv2
import random
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from evogym import EvoWorld, EvoSim, EvoViewer
from evogym import WorldObject
from pathlib import Path

ROBOT_SPAWN_X = 3
ROBOT_SPAWN_Y = 10
ACTUATOR_MIN_LEN = 0.6
ACTUATOR_MAX_LEN = 1.6
FRAME_CYCLE_LEN = 10
NUM_ACTUATORS = 8  #8 for walkbot4billion
NUM_ITERS = 100
MUTATE_RATE = 0.2

ENV_FILENAME = "simple_environment_long.json"
ROBOT_FILENAME = "walkbot4billion.json"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATE_TIME = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
VID_PATH = os.path.join(ROOT_DIR, "videos")

EXPER_DIR = 'score_plots_2/' + ROBOT_FILENAME[:-5] + " " + time.asctime(
)  #directory generated for a single run of the program. Stores outputs.
# [:-5] clips the ".json" off all of the robot names. It is always -5 and it's only used right here.
# Surely that doesn't justify wasting storage on a variable to "de-magic" it? Especially with this helpful explanation?

def create_video(source, fps=50):
    """
    Saves a video from a list of frames

    Parameters:
        source (list): List of cv2 frames.
        output_name (string): Filename of output video.
        vid_path (string): Filepath of output video.
        fps (int): Frames per second of video to save.
    """
    Path(VID_PATH).mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(os.path.join(VID_PATH, DATE_TIME + ".mp4"),
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (source[0].shape[1], source[0].shape[0]))
    for frame in source:
        out.write(frame)
    out.release()
    


def run_rmhc(mode, gens):
    """
    Run a RMHC in evogym. 

    Fitness is defined as the sum of the distance that the 
    robot's center of mass moves in between each step of the simulation.
    The robot morphology is predefined and imported from a json file. 
    It has actuators. The robot's genome is a vector of scalars. 
    At each time step, the x & y coordinates of the robot's center of mass 
    are averaged and this scalar is multiplied by the genome vector. In a loop, 
    for ten frames, this value is multiplied by the first len/2 values of the vector. 
    For the next ten frames, they are multipled by the second len/2. These values are 
    then clipped to between 0.6 and 1.6 and each represent an actuator target length 
    for that simulation step. The genome is mutated by replacing a vector index with 
    a random float between 0 and 1 with a 20% probabillity for each index.
    
    Parameters:
        gens (int): How many generations to run.
        show (bool): If true, renders all simulations. If false, only renders the fittest robot.
    
    Returns:
        (ndarray, fitness): The fittest genome the RMHC finds and its fitness.
    """

    os.mkdir(os.path.join(ROOT_DIR, EXPER_DIR))  #generate special directory for results.

    iters = NUM_ITERS
    genome = np.random.rand(NUM_ACTUATORS * 2)
    best_fitness = run_simulation(iters, genome, mode)

    print("Starting fitness:", best_fitness)
    fitness_by_gen = np.array([])  #array of fitness
    best_fitness_by_gen = np.array([])  #array of fitness (best as of i gen)

    for i in range(gens):

        # Mutate
        mutated_genome = genome.copy()

        mutated_genome = np.array([
            random.random() if random.random() < MUTATE_RATE else x
            for x in mutated_genome
        ])

        new_fitness = run_simulation(iters, mutated_genome, mode)

        fitness_by_gen = np.append(fitness_by_gen, new_fitness)

        # Replace old genome with new genome if it is fitter
        if new_fitness > best_fitness:
            print("Found better after", i, "generations:", new_fitness)
            best_fitness = new_fitness
            genome = mutated_genome

        best_fitness_by_gen = np.append(best_fitness_by_gen, best_fitness)

    # Show fittest genome
    print("Final fitness", best_fitness)
    run_simulation(NUM_ITERS * 5, genome, mode, fittest=True)
    plot_scores(fitness_by_gen, best_fitness_by_gen, gens)

    return (genome, best_fitness)


def run_simulation(iters,
                   genome,
                   mode,
                   fittest=False):  #if fittest, then track action
    """
    Runs a single simulation of a given genome.

    Parameters:
        iters (int): How many iterations to run.
        genome (ndarray): The genome of the robot, which is an 
        array of scalars from the robot's average position to the 
        desired length of the muscles.
        show (bool): Whether or not to display a simulation of the robot.
        fittest (bool):  Whether or not the robot is the fittest if its generation

    Returns:
        float: The fitness of the genome.
    """

    # Create world
    world = EvoWorld.from_json(os.path.join(ROOT_DIR, 'world_data', ENV_FILENAME))

    # Add robot
    robot = WorldObject.from_json(os.path.join(ROOT_DIR, 'world_data', ROBOT_FILENAME))

    world.add_from_array(name='robot',
                         structure=robot.get_structure(),
                         x=ROBOT_SPAWN_X,
                         y=ROBOT_SPAWN_Y,
                         connections=robot.get_connections())

    # Create simulation
    sim = EvoSim(world)
    sim.reset()

    # Set up viewer
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')

    fitness = 0

    action_list = []

    for i in range(iters):

        # Get position of all robot voxels
        pos_1 = sim.object_pos_at_time(sim.get_time(), "robot")

        # Get mean of robot voxels
        com_1 = np.mean(pos_1, 1)

        # Compute the action vector by averaging the avg x & y
        # coordinates and multiplying this scalar by the genome
        action = genome * ((com_1[0] + com_1[1]) / 2)

        # Clip actuator target lengths to be between 0.6 and 1.6 to prevent buggy behavior
        action = np.clip(action, ACTUATOR_MIN_LEN, ACTUATOR_MAX_LEN)

        if i % (FRAME_CYCLE_LEN * 2) < FRAME_CYCLE_LEN:
            action = action[0:NUM_ACTUATORS]
        else:
            action = action[NUM_ACTUATORS:(NUM_ACTUATORS * 2)]

        # Set robot action to the action vector. Each actuator corresponds to a vector
        # index and will try to expand/contract to that value
        sim.set_action('robot', action)

        # Execute step
        sim.step()

        #action tracking for fittest
        if fittest:
            print(action)
            action_list.append(action)

        # Get robot position after the step
        pos_2 = sim.object_pos_at_time(sim.get_time(), "robot")

        # Compute reward, how far the robot moved in that time step
        com_2 = np.mean(pos_2, 1)
        reward = com_2[0] - com_1[0]
        fitness += reward

    viewer.close()

    if fittest:
        action_array = np.asarray(action_list)
        plot_action(action_array)
        run_simulation_from_action(
            iters, action_array, mode, True)  #Yeah, this is a hack, I know. I'm sorry.

    return fitness


def run_simulation_from_action(iters,
                               given_actions,
                               mode,
                               fittest=False):  #if fittest, then track action
    """
    Runs a single sped-up simulation of a given collection of action arrays.

    Parameters:
        iters (int): How many iterations to run.
        given_actions (ndarray): action arrays of the existing robot. to be modified for example.
        show (bool): Whether or not to display a simulation of the robot.
        fittest (bool):  Whether or not the robot is the fittest if its generation

    Returns:
        float: The fitness of the genome. (though this value is not used or updated here.)
    """

    # Create world
    # Create world
    world = EvoWorld.from_json(os.path.join(ROOT_DIR, 'world_data', ENV_FILENAME))

    # Add robot
    robot = WorldObject.from_json(os.path.join(ROOT_DIR, 'world_data', ROBOT_FILENAME))

    world.add_from_array(name='robot',
                         structure=robot.get_structure(),
                         x=ROBOT_SPAWN_X,
                         y=ROBOT_SPAWN_Y,
                         connections=robot.get_connections())

    # Create simulation
    sim = EvoSim(world)
    sim.reset()

    # Modify action array
    # Take take every other element, then double the remaining.
    # "speeds up" robot's actions.
    halved = given_actions[::2]
    sped_up_action_array = np.concatenate(
        (halved, halved)
    )  #glues two halves together so the total number of action arrays remains the same.

    # Set up viewer
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')

    video_frames = []

    fitness = 0

    action_list = []

    for i in range(iters):

        action = sped_up_action_array[i - 1]

        # Set robot action to the action vector. Each actuator corresponds to a vector
        # index and will try to expand/contract to that value
        sim.set_action('robot', action)

        # Execute step
        sim.step()

        #action tracking for fittest
        if fittest:
            print(action)
            action_list.append(action)

        if mode == "s":
            viewer.render('screen', verbose=True)
        elif mode == "v":
            video_frames.append(viewer.render(verbose=False, mode="rgb_array"))
        elif mode == "b":
            viewer.render('screen', verbose=True)
            video_frames.append(viewer.render(verbose=False, mode="rgb_array"))

    viewer.close()

    if mode in ["v", "b"]:
        create_video(video_frames)

    if fittest:
        action_array = np.asarray(action_list)
        plot_action(action_array, sped=True)

    return fitness


def plot_scores(fit_func_scores, fit_func_scores_best, gen_number):
    '''
    Plots the scores of the robot on the fitness function.
    Plots both each generation and the best by "x" generation.

    Parameters:
        fit_func_scores (ndarray): Contains the scores of each generation.
        fit_func_scores_best (ndarray): Contains the best scores at or before each generation.
        gen_number (int): the number of generations in this simulation.

    Returns:
        A matplotlib plot output to a directory. Plot shows the fitness function score over the generations.
    '''

    #make plot
    gen_array = np.arange(1, gen_number + 1, 1)
    plottitle = ROBOT_FILENAME + " scores"
    plt.title(plottitle)
    plt.ylabel("scores")
    plt.xlabel("generations")
    plt.plot(gen_array, fit_func_scores, label="at gen")
    plt.plot(gen_array, fit_func_scores_best, label="best by gen")
    plt.legend(loc='upper center')
    plt.savefig(os.path.join(ROOT_DIR, EXPER_DIR + "/" + plottitle + '.png'))
    plt.close()


def plot_action(action_arrays, sped=False):
    '''
    Plots the voxel action arrays at each step of a simulation

    Parameters:
        action_arrays (ndarray): Contains more ndarrays, one for each action array. One action array per simulator step. 
        sped (bool): If this is the "sped up" version of the actions or not.

    Returns:
        Several matplotlib plot outputs to a directory. One with all actuator voxels and then one per voxel.
        Also outputs an excel spreadsheet of all of the actuators' actions. 
    '''

    stepcount_array = np.arange(1, len(action_arrays) + 1, 1)
    plottitle = ROBOT_FILENAME + " actions"
    plt.title(plottitle)
    plt.ylabel("target length")
    plt.xlabel("steps")

    voxels_list = []
    for i in range(len(action_arrays[0])):  #for each voxel
        voxel_action = np.array([])
        for j in range(len(action_arrays)):  #for each step
            voxel_action = np.append(
                voxel_action, action_arrays[j]
                [i])  #I use both i and j, so this is okay, right?
        #plot_voxel(stepcount_array, voxel_action, i)
        voxels_list.append(voxel_action)
        plt.plot(stepcount_array, voxel_action, label="voxel: " + str(i))

    plt.legend(loc='upper center')
    if sped:
        plt.savefig(os.path.join(ROOT_DIR, EXPER_DIR + "/" + plottitle + ' (sped).png'))
    else:
        plt.savefig(os.path.join(ROOT_DIR, EXPER_DIR + "/" + plottitle + '.png'))
    plt.close()

    #Save voxel data for later examination
    df = pd.DataFrame(voxels_list)
    if sped:
        df.to_excel(os.path.join(ROOT_DIR, EXPER_DIR + "/" + plottitle + ' (sped).xlsx'),
                    index=False)
    else:
        df.to_excel(os.path.join(ROOT_DIR, EXPER_DIR + "/" + plottitle + '.xlsx'), index=False)

    #plot individual voxels
    for i in range(len(voxels_list)):  #I USE i!!!!!
        plot_voxel(stepcount_array, voxels_list[i], i, sped)


def plot_voxel(stepcount_array, voxel_action, voxel_num, sped=False):
    '''
    Plots the voxel action arrays at each step of a simulation

    Parameters:
        stepcount_array (ndarray): An array with the number of steps from 1 to n. 
        voxel_action (ndarray): An array with the actuator's actions at each step. 
        voxel_num: The actuator number. Used to differentiate this actuator from the others.
        sped (bool): If this is the "sped up" version of the actions or not.

    Returns:
        A matplotlib plot output to a directory. Plot shows the actuator's action over the course of the simulation.
    '''
    plottitle = ROBOT_FILENAME + " voxel: " + str(voxel_num)
    plt.title(plottitle)
    plt.ylabel("target length")
    plt.xlabel("steps")
    plt.plot(stepcount_array, voxel_action, label="voxel: " + str(voxel_num))
    plt.legend(loc='upper center')
    if sped:
        plt.savefig(os.path.join(ROOT_DIR, EXPER_DIR + "/" + plottitle + ' (sped).png'))
    else:
        plt.savefig(os.path.join(ROOT_DIR, EXPER_DIR + "/" + plottitle + '.png'))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--mode',  # headless, screen, video, both h, s, v, b
        help='s-screen, v-video, b-both',
        default="b")
    parser.add_argument('--gens',
                        type=float,
                        default=1000,
                        help='sigma value for cma-es')
    args = parser.parse_args()

    run_rmhc(args.mode, args.gens)

# TWO PROOFS:
# "can" action and replay it sped up vs normal. Robot should act totally different
# Robot with "dead" actuators should behave different from normal healthy robot.
