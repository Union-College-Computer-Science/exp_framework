"""
Given a genome, runs a simulation of a walking robot in evogym, using an SNN controlled robot,
providing a fitness score corresponding to how far the robot walked.

Author: Thomas Breimer, James Gaskell
January 29th, 2025
"""

import os
import csv
import itertools
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from evogym import EvoWorld, EvoSim, EvoViewer
from evogym import WorldObject
from snn_sim.robot.morphology import Morphology
from snn_sim.snn.snn_controller import SNNController

# Simulation constants
ROBOT_SPAWN_X = 3
ROBOT_SPAWN_Y = 1
ACTUATOR_MIN_LEN = 0.6
ACTUATOR_MAX_LEN = 1.6
NUM_ITERS = 1000
FPS = 50
MODE = "v" # "headless", "screen", or "video"

FITNESS_OFFSET = 100
NUM_ACTUATORS = 7

# Files
ENV_FILENAME = "big_platform.json"
ROBOT_FILENAME = "bestbot.json"
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
DATE_TIME = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

def create_header():

    csv_path = (os.path.join(os.path.dirname(THIS_DIR), "plot_firing/data", f"{DATE_TIME}.csv"))

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    csv_header = (["Actuator:" f"{i}" for i in range(NUM_ACTUATORS)])

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header) 

    return(csv_path)


def plot_firing_action(action, csv_path):
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        flat_action = [item[0] for item in action]
        writer.writerow(flat_action)
    


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
                          fps, (source[0].shape[1], source[0].shape[0]))
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

def run(iters, genome, mode, file, vid_name=None, vid_path=None):
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

    robot_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'robot', 'world_data', ROBOT_FILENAME)

    snn_controller = SNNController(2, 2, 1, robot_config=robot_file_path)
    snn_controller.set_snn_weights(genome)

    log = []

    for _ in range(iters):
        # Get point mass locations
        raw_pm_pos = sim.object_pos_at_time(sim.get_time(), "robot")

        # Get distances to the corners
        corner_distances = morphology.get_corner_distances(raw_pm_pos)

        # Feed snn and get outputs
        action = snn_controller.get_lengths(corner_distances)
        
        plot_firing_action(action, file)

        # Clip actuator target lengths to be between 0.6 and 1.6 to prevent buggy behavior
        action = np.clip(action, ACTUATOR_MIN_LEN, ACTUATOR_MAX_LEN)
        log.append(action)

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

    log_iter = []
    for x in log:
        log_iter.append(list(itertools.chain.from_iterable(x)))

    viewer.close()

    # Get robot point mass position position afer sim has run
    final_raw_pm_pos = sim.object_pos_at_time(sim.get_time(), "robot")

    fitness = np.mean(final_raw_pm_pos[0]) - np.mean(init_raw_pm_pos[0])

    if mode in ["v", "b"]:
        create_video(video_frames, vid_name, vid_path, FPS)

    return FITNESS_OFFSET - fitness, log_iter # Turn into a minimization problem