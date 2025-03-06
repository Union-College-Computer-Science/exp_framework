"""
Representation of an evogym robot.

Authors: Thomas Breimer, Matthew Meek
February 21st, 2025
"""

import os
import numpy as np
from evogym import WorldObject
from rsnn_sim.robot.actuator import Actuator

ROBOT_SPAWN_X = 0
ROBOT_SPAWN_Y = 10
ENV_FILENAME = "simple_environment.json"


class Morphology:
    """
    Our own internal representation of an evogym robot.
    """

    def __init__(self, filename: str):
        """
        Given an evogym robot file, constructs a robot morphology.

        Parameters:
            filename (str): Filename of the robot .json file.
        """

        self.robot_filepath = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "world_data",
            filename)
        self.structure = self.get_structure(self.robot_filepath)
        self.actuators = self.create_actuator_voxels(self.structure)

    def get_structure(self, robot_filepath: str) -> np.ndarray:
        """
        Return the robot’s structure matrix.

        Parameters:
            robot_filepath (str): The filename of the robot to get.

        Returns:
            np.ndarray: (n, m) array specifing the voxel structure of the object.
        """

        # Get robot structure
        robot = WorldObject.from_json(robot_filepath)
        return robot.get_structure()

    def create_actuator_voxels(self, structure: np.ndarray) -> list:
        """
        Given a robot structure, creates vertices. Also sets the top left
        and bottom right indicies.

        Parameters:
            structure (np.ndarray): array specifing the voxel structure of the object.

        Returns:
            list: A list of actuator objects.
        """

        # Evogym assigns point mass indices by going through the structure array
        # left to right, top to bottom. The first voxel it sees, it assigns its
        # top left point mass to index zero, top right point mass to index one,
        # bottom left point mass to index two, and bottom right point mass to
        # index three. This pattern continues, expect that any point masses that
        # are shared with another voxel and have already been seen are not added to
        # the point mass array. This script goes through this process, constructing
        # the point mass array and identifying shared point masses to create correct
        # actuator objects.

        # To return, will contain Actuator objects
        actuators = []

        # List of tuples (x, y) corresponding to initial point mass positions and index
        # within this list corresponding the their index when calling robot.get_pos()
        self.point_masses = []

        # Dimensions of the robot
        height = len(structure)
        #length = len(structure[0])

        # Will be the coordinates of the top left point mass of ther current voxel.
        top_y = height
        left_x = 0

        # Follows a similar pattern to point masses, top right actuator is zero,
        # and increments going left to right then top to bottom down the grid
        actuator_action_index = 0

        for row in structure:
            for voxel_type in row:
                if not voxel_type == 0:  # Don't add empty voxels

                    right_x = left_x + 1
                    bottom_y = top_y - 1

                    # Check if top left point mass already in point_masses
                    if (left_x, top_y) in self.point_masses:
                        # If so, find index will be the index of where it already is in the array
                        top_left_index = self.point_masses.index(
                            (left_x, top_y))
                    else:
                        # Else, we make a new point mass position
                        top_left_index = len(self.point_masses)
                        self.point_masses.append((left_x, top_y))

                    # Repeat for top right point mass
                    if (right_x, top_y) in self.point_masses:
                        top_right_index = self.point_masses.index(
                            (right_x, top_y))
                    else:
                        top_right_index = len(self.point_masses)
                        self.point_masses.append((right_x, top_y))

                    # And for bottom left point mass
                    if (left_x, bottom_y) in self.point_masses:
                        bottom_left_index = self.point_masses.index(
                            (left_x, bottom_y))
                    else:
                        bottom_left_index = len(self.point_masses)
                        self.point_masses.append((left_x, bottom_y))

                    # And finally bottom right
                    if (right_x, bottom_y) in self.point_masses:
                        bottom_right_index = self.point_masses.index(
                            (right_x, bottom_y))
                    else:
                        bottom_right_index = len(self.point_masses)
                        self.point_masses.append((right_x, bottom_y))

                    # Voxel types 3 and 4 are actuators.
                    # Don't want to add voxel if its not an actuator
                    if voxel_type in [3, 4]:
                        pmis = np.array([
                            top_left_index, top_right_index, bottom_left_index,
                            bottom_right_index
                        ])
                        actuator_obj = Actuator(actuator_action_index,
                                                voxel_type, pmis)
                        actuators.append(actuator_obj)
                        actuator_action_index += 1

                left_x += 1

            top_y -= 1
            left_x = 0

        self.top_left_corner_index = 0
        self.bottom_right_corner_index = len(self.point_masses) - 1

        return actuators

    def get_corner_distances(self, pm_pos: list) -> tuple:
        """
        Given the list of robot point mass coordinates generated from sim.object_pos_at_time(),
        returns an list of an lists where each top level list corresponds to a an actuator voxel,
        the the sublist contains the distance to the [top left corner, bottom right corner].
        
        Parameters:
            pm_pos (list): A list with the first element being a np.ndarray containing all
                           point mass x positions, and second element containig all point mass
                           y positions.
        
        Returns:
            list: A tuple of the distances to the top left point mass and bottom right point mass.
        """

        actuator_distances = []

        for actuator in self.actuators:
            actuator_distances.append(
                actuator.get_distances_to_corners(
                    pm_pos, self.top_left_corner_index,
                    self.bottom_right_corner_index))

        return actuator_distances
