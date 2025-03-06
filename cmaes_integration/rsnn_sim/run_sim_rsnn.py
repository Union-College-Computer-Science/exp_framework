"""
Given a genome, runs a simulation of a walking robot in evogym, using an SNN controlled robot,
providing a fitness score corresponding to how far the robot walked.

Author: Thomas Breimer, James Gaskell, Guy Tallent
January 29th, 2025
"""

# pylint struggling to import evogym
# pylint: disable = import-error
# pylint: disable = no-member

import os
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from evogym import EvoWorld, EvoSim, EvoViewer
from evogym import WorldObject
from rsnn_sim.robot.morph2 import Morphology
from rsnn_sim.snn.snn_controller import SNNController

# Simulation constants
ROBOT_SPAWN_X = 3
ROBOT_SPAWN_Y = 1
ACTUATOR_MIN_LEN = 0.6
ACTUATOR_MAX_LEN = 1.6
NUM_ITERS = 200
FPS = 50
MODE = "v" # "headless", "screen", or "video"

FITNESS_OFFSET = 10
SNN_CONTEXT_LENGTH = 5 # Number of previous actions to consider in Neural Network

# Files
ENV_FILENAME = "big_platform.json"
ROBOT_FILENAME = "bestbot.json"
THIS_DIR = os.path.dirname(os.path.realpath(__file__))

def create_video(source, output_name, vid_path, fps=FPS):
    """
    Saves a video from a list of frames

    Parameters:
        source (list): List of cv2 frames.
        output_name (string): Filename of output video.
        vid_path (string): Filepath of output video.
        fps (int): Frames per second of video to save.
    """

    Path(vid_path).mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(os.path.join(vid_path, output_name + ".mp4"),
                           cv2.VideoWriter_fourcc(*'mp4v'),
                           fps,
                           (source[0].shape[1],
                            source[0].shape[0]))

    for frame in source:
        out.write(frame)
    out.release()

def group_list(flat_list: list, n: int) -> list:
    """
    Groups flat_array into a list of list of size n.

    Parameters:
        flat_list (list): List to groups.
        n: (int): Size of sublists.
    
    Returns:
        list: Grouped list.
    """
    return [list(flat_list[i:i+n]) for i in range(0, len(flat_list), n)]

def run(iters, genome, mode, vid_name=None, vid_path=None):
    """
    Runs a single simulation of a given genome.

    Parameters:
        iters (int): How many iterations to run.
        genome (ndarray): The genome of the robot.
        mode (string): How to run the simulation. 
                       "h" runs without any video or visual output.
                       "v" outputs the simulation as a video in the "./videos folder.
                       "s" shows the simulation on screen as a window.
                       "b: shows the simulation on a window and saves a video.
        vid_name (string): If mode is "v" or "b", this is the name of the saved video.
        vid_path (string): If mode is "v" or "b", this is the path the video will be saved.
    Returns:
        float: The fitness of the genome.
    """

    prev_action = []

    # Create world
    world = EvoWorld.from_json(os.path.join(THIS_DIR, 'robot', 'world_data', ENV_FILENAME))

    # Add robot
    robot = WorldObject.from_json(os.path.join(THIS_DIR, 'robot', 'world_data', ROBOT_FILENAME))

    world.add_from_array(
        name='robot',
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

    video_frames = []

    # Get position of all robot point masses
    init_raw_pm_pos = sim.object_pos_at_time(sim.get_time(), "robot")

    morphology = Morphology(ROBOT_FILENAME)

    file_path = os.path.dirname(os.path.abspath(__file__))
    robot_file_path = os.path.join(file_path, 'robot', 'world_data', ROBOT_FILENAME)

    snn_controller = SNNController(4+SNN_CONTEXT_LENGTH, 4+SNN_CONTEXT_LENGTH, 1, robot_config=robot_file_path)
    snn_controller.set_snn_weights(genome)

    # put all the point masses (pm_coords) into pairs of x, y coordinates
    pm_pairs = list(zip(init_raw_pm_pos[0], init_raw_pm_pos[1]))

    # creates a dictionary that maps each pm_pair to its index
    # used later to convert from coords to their indices
    point_index_map = {coord: idx for idx, coord in enumerate(pm_pairs)}
    # print(point_index_map)

    # iterate through the pm coord pairs and group the indices based on y coord
    grouped_coords = defaultdict(list)
    for idx, (_, y) in enumerate(pm_pairs):
        grouped_coords[y].append(idx)  # Append index instead of (x, y)
    # print(grouped_coords)

    for i in range(iters):
        if i % 12 == 0:
            # Get point mass locations
            raw_pm_pos = sim.object_pos_at_time(sim.get_time(), "robot")
            # print(raw_pm_pos)

            # maps pms to active voxels and corners
            voxels, corners = morphology.map_pm_to_active_voxels(raw_pm_pos, grouped_coords, robot)

            # print (voxels)
            # print (corners)

            # Get distances to the corners
            corner_distances = morphology.get_corner_distances(raw_pm_pos)
            # print(corner_distances)

            # Feed snn and get outputs
            action = snn_controller.get_lengths(corner_distances, prev_action)

            # Clip actuator target lengths to be between 0.6 and 1.6 to prevent buggy behavior
            action = np.clip(action, ACTUATOR_MIN_LEN, ACTUATOR_MAX_LEN)

            if len(prev_action) < SNN_CONTEXT_LENGTH:
                action = 1/action
                prev_action.append(action)
            elif len(prev_action) == SNN_CONTEXT_LENGTH:
                action = 1/action
                prev_action.pop(0)
                prev_action.append(action)
        

            # Set robot action to the action vector. Each actuator corresponds to a vector
            # index and will try to expand/contract to that value
            sim.set_action('robot', action)

            # Execute step
        sim.step()

        if mode == "v":
            video_frames.append(viewer.render(verbose=False, mode="rgb_array"))
        elif mode == "s":
            viewer.render(verbose=True, mode="screen")
        elif mode == "b":
            viewer.render(verbose=True, mode="screen")
            video_frames.append(viewer.render(verbose=False, mode="rgb_array"))

    viewer.close()

    # Get robot point mass position position afer sim has run
    final_raw_pm_pos = sim.object_pos_at_time(sim.get_time(), "robot")

    fitness = np.mean(final_raw_pm_pos, 1)[0] - np.mean(init_raw_pm_pos, 1)[0]
    
    if np.mean(init_raw_pm_pos, 1)[1] - np.mean(final_raw_pm_pos, 1)[1] > 0:
        fitness = 0

    if mode in ["v", "b"]:
        create_video(video_frames, vid_name, vid_path, FPS)

    return FITNESS_OFFSET - fitness # Turn into a minimization problem
