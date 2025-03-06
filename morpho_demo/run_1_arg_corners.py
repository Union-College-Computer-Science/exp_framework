"""
Simple RMHC of walking robot from scratch in evogym

Author: Thomas Breimer, Matthew Meek
January 22nd, 2025
"""

import os
import random
import time
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from evogym import EvoWorld, EvoSim, EvoViewer
from evogym import WorldObject
from special_classes import corner as corn

ROBOT_SPAWN_X = 3
ROBOT_SPAWN_Y = 10
ACTUATOR_MIN_LEN = 0.6
ACTUATOR_MAX_LEN = 1.6
FRAME_CYCLE_LEN = 10
#NUM_ACTUATORS = 8  #8 for walkbot4billion
#NUM_ITERS = 100
MUTATE_RATE = 0.2

ENV_FILENAME = "simple_environment_long.json"
#ROBOT_FILENAME = "walkbot4billion.json"
#GENERATIONS = 1500

#EXPER_DIR = 'score_plots/' + ROBOT_FILENAME[:-5] + " " + time.asctime(
#)  #directory generated for a single run of the program. Stores outputs.


def retrieve_actuator_count(robot_filename):
    '''
    Retrieves the number of actuator voxels the given robot has.

    Parameters:
        robot_filename (str): Location of robot json.

    Returns:
        An int: the number of actuators in the robot.
    '''
    jsonf = open(os.path.join(os.path.dirname(__file__) + "/world_data", robot_filename))
    data = json.load(jsonf)
    actuator_count = 0
    key_for_robot = list(data["objects"].keys())[0] # No, I'm not happy about it, either.
    # The robot's key in the json dicts can vary because the design tool is stupid.
    # So this is a fix.
    for voxel in data["objects"][key_for_robot]["types"]:
        if voxel in (3, 4): # 3 and 4 represent horizontal and vertical actuators.
            actuator_count += 1

    return actuator_count


def run_rmhc(gens, iters, robot_filename, exper_dir, show=True):
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
        iters (int): Iterations per generation.
        robot_filename (str): Location of robot json.
        exper_dir (str): Location of experiment directory.
        show (bool): If true, renders all simulations. If false, only renders the fittest robot.
    
    Returns:
        (ndarray, fitness): The fittest genome the RMHC finds and its fitness.
    """

    os.mkdir(os.path.join(os.path.dirname(__file__),exper_dir))   #generate special directory for results.

    corners = find_corners(robot_filename)
    print("Muh corners: " + str(corners))

    num_actuators = retrieve_actuator_count(robot_filename)
    genome = np.random.rand(num_actuators * 2)
    best_fitness = run_simulation(iters, genome, robot_filename,
                                  corners, exper_dir, show)

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

        new_fitness = run_simulation(iters, mutated_genome, robot_filename,
                                     corners, exper_dir, show)

        fitness_by_gen = np.append(fitness_by_gen, new_fitness)

        # Replace old genome with new genome if it is fitter
        if new_fitness > best_fitness:
            print("Found better after", i, "generations:", new_fitness)
            best_fitness = new_fitness
            genome = mutated_genome

        best_fitness_by_gen = np.append(best_fitness_by_gen, best_fitness)

    # Show fittest genome
    print("Final fitness", best_fitness)
    run_simulation(iters * 5, genome, robot_filename, corners, exper_dir, fittest=True)
    plot_scores(fitness_by_gen, best_fitness_by_gen, gens, robot_filename, exper_dir)

    return (genome, best_fitness)


def run_simulation(iters,
                   genome,
                   robot_filename,
                   corners,
                   exper_dir,
                   show=True,
                   fittest=False):  #if fittest, then track action
    """
    Runs a single simulation of a given genome.

    Parameters:
        iters (int): How many iterations to run.
        genome (ndarray): The genome of the robot, which is an 
        array of scalars from the robot's average position to the 
        desired length of the muscles.
        robot_filename (str): Location of robot json.
        show (bool): Whether or not to display a simulation of the robot.
        fittest (bool):  Whether or not the robot is the fittest if its generation

    Returns:
        float: The fitness of the genome.
    """

    # Create world
    world = EvoWorld.from_json(os.path.join(os.path.dirname(__file__) + '/world_data', ENV_FILENAME))

    # Add robot
    robot = WorldObject.from_json(os.path.join(os.path.dirname(__file__) + "/world_data", robot_filename))

    world.add_from_array(name='robot',
                         structure=robot.get_structure(),
                         x=ROBOT_SPAWN_X,
                         y=ROBOT_SPAWN_Y,
                         connections=robot.get_connections())

    #get actuators
    num_actuators = retrieve_actuator_count(robot_filename)

    # Create simulation
    sim = EvoSim(world)
    sim.reset()

    # Set up viewer
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')

    fitness = 0

    action_list = []

    for i in range(iters):

        # Get position of all robot point masses
        pos_1 = sim.object_pos_at_time(sim.get_time(), "robot")

        # Get mean of robot point masses
        com_1 = np.mean(pos_1, 1)


        # CORNERS TELEMETRY STUFF
        # SNN SQUAD, YOUR DATA IS HERE
        # CHANGE THE PRINT TO BE WHATEVER YOUR STUFF NEEDS!
        distances = get_all_distances(pos_1, corners)
        print("Corners dists at iter " + str(i) + " :\n" + str(distances) +"\n")
        # and also probably mess with the genome stuff or something idk
        # /CORNERS TELEMETRY STUFF


        # Compute the action vector by averaging the avg x & y
        # coordinates and multiplying this scalar by the genome
        action = genome * ((com_1[0] + com_1[1]) / 2)

        # Clip actuator target lengths to be between 0.6 and 1.6 to prevent buggy behavior
        action = np.clip(action, ACTUATOR_MIN_LEN, ACTUATOR_MAX_LEN)

        if i % (FRAME_CYCLE_LEN * 2) < FRAME_CYCLE_LEN:
            action = action[0:num_actuators]
        else:
            action = action[num_actuators:(num_actuators * 2)]

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

        if show:
            viewer.render('screen', verbose=True)
    viewer.close()

    if fittest:
        action_array = np.asarray(action_list)
        plot_action(action_array, robot_filename, exper_dir)

    return fitness


def plot_action(action_arrays, robot_filename, exper_dir):
    '''
    Plots the voxel action arrays at each step of a simulation

    Parameters:
        action_arrays (ndarray): Contains more ndarrays, one for each action array.
        One action array per simulator step.
        robot_filename (str): Location of robot json.
        exper_dir (str): Location of experiment directory. 

    Returns:
        Several matplotlib plot outputs to a directory.
        One with all actuator voxels and then one per voxel.
        Also outputs an excel spreadsheet of all of the actuators' actions. 
    '''

    stepcount_array = np.arange(1, len(action_arrays) + 1, 1)
    plottitle = robot_filename + " actions"
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
        voxels_list.append(voxel_action)
        plt.plot(stepcount_array, voxel_action, label="voxel: " + str(i))

    plt.legend(loc='upper center')
    plt.savefig(os.path.join(exper_dir, plottitle + '.png'))
    plt.close()

    #Save voxel data for later examination
    df = pd.DataFrame(voxels_list)
    df.to_csv(os.path.join(exper_dir, plottitle + '.csv'), index=False)

    #plot individual voxels
    for i in range(len(voxels_list)):  #I USE i!!!!!
        plot_voxel(stepcount_array, voxels_list[i], i, robot_filename, exper_dir)


def plot_voxel(stepcount_array, voxel_action, voxel_num, robot_filename, exper_dir):
    '''
    Plots the voxel action arrays at each step of a simulation

    Parameters:
        stepcount_array (ndarray): An array with the number of steps from 1 to n. 
        voxel_action (ndarray): An array with the actuator's actions at each step. 
        voxel_num: The actuator number. Used to differentiate this actuator from the others.
        robot_filename (str): Location of robot json.
        exper_dir (str): Location of experiment directory.

    Returns:
        A matplotlib plot output to a directory.
        Plot shows the actuator's action over the course of the simulation.
    '''
    plottitle = robot_filename + " voxel: " + str(voxel_num)
    plt.title(plottitle)
    plt.ylabel("target length")
    plt.xlabel("steps")
    plt.plot(stepcount_array, voxel_action, label="voxel: " + str(voxel_num))
    plt.legend(loc='upper center')
    plt.savefig(os.path.join(exper_dir, plottitle + '.png'))
    plt.close()

def plot_scores(fit_func_scores, fit_func_scores_best, gen_number, robot_filename, exper_dir):
    '''
    Plots the scores of the robot on the fitness function.
    Plots both each generation and the best by "x" generation.

    Parameters:
        fit_func_scores (ndarray): Contains the scores of each generation.
        fit_func_scores_best (ndarray): Contains the best scores at or before each generation.
        gen_number (int): the number of generations in this simulation.
        robot_filename (str): Location of robot json.
        exper_dir (str): Location of experiment directory.

    Returns:
        A matplotlib plot output to a directory.
        Plot shows the fitness function score over the generations.
    '''

    #make plot
    gen_array = np.arange(1, gen_number + 1, 1)
    plottitle = robot_filename + " scores"
    plt.title(plottitle)
    plt.ylabel("scores")
    plt.xlabel("generations")
    plt.plot(gen_array, fit_func_scores, label="at gen")
    plt.plot(gen_array, fit_func_scores_best, label="best by gen")
    plt.legend(loc='upper center')
    plt.savefig(os.path.join(exper_dir, plottitle + '.png'))
    plt.close()

    #save scores for later
    df = pd.DataFrame((fit_func_scores,fit_func_scores_best))
    df.to_csv(os.path.join(exper_dir, plottitle + '.csv'), index=False)


def find_corners(robot_filename):
    '''
    Finds the corners, returns them in an array of four corner objects.

    WARNING:
    This function expects a robot whose outermost corners are at the 
    minimum and maximum x and y values. 
    IT WILL BREAK IF THIS IS NOT THE CASE!!!

    Parameters:
        robot_filename (string): name of the json file the robot lives in.

    Returns:
        An ndarray of four corner objects. Each represents an outer corner.
    '''
    # Stupid little hack but it works like magic.
    # We create a 0-time "simulation" to get coords for point masses.
    # Using those, we can figure out the indicies for the corners.
    # Create world
    world = EvoWorld.from_json(os.path.join(os.path.dirname(__file__) + '/world_data', ENV_FILENAME))

    # Add robot
    robot = WorldObject.from_json(os.path.join(os.path.dirname(__file__) + "/world_data", robot_filename))
    world.add_from_array(name='robot',
                         structure=robot.get_structure(),
                         x=ROBOT_SPAWN_X,
                         y=ROBOT_SPAWN_Y,
                         connections=robot.get_connections())
    sim = EvoSim(world)
    sim.reset()
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')
    xy_coords = sim.object_pos_at_time(sim.get_time(), "robot")
    viewer.close()

    # xy_coords is an ndarray of ndarrays
    # first array is x, second array is y.
    x_min = xy_coords[0].min()
    y_min = xy_coords[1].min()
    x_max = xy_coords[0].max()
    y_max = xy_coords[1].max()

    top_right = corn.Corner(find_pm_index(xy_coords, x_max, y_max))
    top_left = corn.Corner(find_pm_index(xy_coords, x_min, y_max))
    bottom_right = corn.Corner(find_pm_index(xy_coords, x_max, y_min))
    bottom_left = corn.Corner(find_pm_index(xy_coords, x_min, y_min))

    toreturn = np.array([top_right, top_left, bottom_right, bottom_left])

    return toreturn


def find_pm_index(xy_coords, x_target, y_target):
    '''
    Finds the index of the point mass at the given x and y coords.

    Parameters:
        xy_coords (ndarray): array containg all of the point mass coords.
        x_target (int): x coord of desired point mass.
        y_target (int): y coord of desired point mass

    Returns:
        int index in xy_coords of the point mass with the desired x and y coords
    '''
    for i in range(len(xy_coords[0])): # i is the point, so this isn't a list comp thing.
        if xy_coords[0][i] == x_target and  xy_coords[1][i] == y_target:
            return i
    raise ValueError("No point mass with target coords.")


def get_all_distances(xy_coords, corners):
    '''
    Gets all of the distances between all corners.
    Excludes 0s (distance from corner to itself) and repeats. 

    Parameters:
        xy_coords (ndarray): array containg all of the point mass coords.
        corners (ndarray): An array of corner objects.

    Returns:
        A 1D ndarray of floats, the distances between the corners. 
        Array should be of len (n-1)th triangular number, where n is the len of corners
        (that is, the number of corners in corners)

    '''

    all_distances = []

    # Nested for loops?!
    # BARF!
    for i in range(len(corners)): #use i to avoid inserting duplicate
        c_distances = corners[i].get_corner_distances(xy_coords, corners)
        #print(str(corners[i]) + " to all: " + str(c_distances))
        for distance in c_distances[i+1:]: #excludes 0s and pre-existing values
            all_distances.append(distance)

    return np.array(all_distances)


if __name__ == "__main__":

    #args
    parser = argparse.ArgumentParser(description="Arguments for simple experiment.")
    parser.add_argument("--robot_filename", default="bestbot.json", type=str,
                        help="The json file of the robot you want to run the experiment with.")
    parser.add_argument("--iters", default=100, type=int,
                        help="number of iterations per generation.")
    parser.add_argument("--gens", default=1500, type=int,
                        help="number of generations.")

    args = parser.parse_args()

    exper_dir_out = os.path.join('score_plots',
                             args.robot_filename[:-5] + " " + time.asctime())
    genome_out, best_fitness_out = run_rmhc(args.gens,
                                    args.iters, args.robot_filename, exper_dir_out, show=False)

    # python run_1_arg_corners.py bestbot.json
