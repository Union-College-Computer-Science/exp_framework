"""
Representation of an evogym robot.

Authors: Thomas Breimer, Matthew Meek, Guy Tallent
February 21st, 2025
"""

import os
import numpy as np
from evogym import WorldObject
import json
from collections import defaultdict
from rsnn_sim.robot.actuator2 import Actuator

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
    
    def map_pm_to_active_voxels(self, coords, grouped_coords, robot):
        """
        Maps point masses (pms) to active voxels and corners for a robot.
        
        Parameters:
            coords: a 2D array containing x and y coords of pms.
            grouped_coords: a dict that groups pms by their y coords. Keys are the y coords and values
            are indicies that point to the pms.
            robot: robot being used in the simulation.

        Returns:
            filtered_voxels: a list of tuples that contains the voxel type, 3 or 4 which
            correspond to active voxels. A list of indices that represent the pms that 
            form the voxel.
            corner_voxels: a list of indices that correspond to the robots four corners,
            the top_left, top_right, bottom_left and bottom_right.

        """

        # get robot structure
        robot_structure = robot.get_structure()

        num_rows = robot_structure.shape[0]     # number of rows in robot structure

        # get positions of all point masses at the current time step
        pm_coords = coords

        # maps each pm_pair to its index - used later to convert from coords to their indices
        point_index_map = {coord: idx for idx, coord in enumerate(zip(pm_coords[0], pm_coords[1]))}

        # groups pms indices based on their y coords
        grouped_indices = defaultdict(list)
        for y in grouped_coords:
            for idx in grouped_coords[y]:
                grouped_indices[y].append(idx)

        # groups pm coords based on their y coords
        grouped_coords_by_y = defaultdict(list)
        for y in grouped_coords:
            for idx in grouped_coords[y]:
                grouped_coords_by_y[y].append((pm_coords[0][idx], pm_coords[1][idx]))

        # creates ndarray where n is the different possible y coords. Each array of indices 
        # is then sorted least to greatest based on x coord
        sorted_groups_indices = [
            sorted(grouped_indices[y], key=lambda idx: pm_coords[0][idx])  # sort by x-coordinates
            for y in sorted(grouped_coords.keys(), reverse=True)  # sort by y-coordinates (highest y first)
        ]

        # creates ndarray where n is the different possible y coords. Each array of coords 
        # is then sorted least to greatest based on x coord
        sorted_groups_coords = [
            sorted(grouped_coords_by_y[y], key=lambda coord: coord[0])  # sort by x coords within each group
            for y in sorted(grouped_coords.keys(), reverse=True)  # sort by y coords (highest y first)
        ]

        # list to store the four pms for each voxel
        grouped_voxels = []
        j = 0  # row index

        for j in range(num_rows):
            if j >= len(sorted_groups_indices) - 1:
                break

            top_row = sorted_groups_coords[j]       # pms in current row (contains the top left and right pms for each voxel)
            bottom_row = sorted_groups_coords[j + 1]  # pms in the next row (contains the bottom left and right pms for each voxel)

            i = 0   # iterates through top row
            v = 0   # iterates through bottom row
            while i < len(top_row) - 1:
                if robot_structure[j, i] != 0:      # check to see if voxel at current position has type of non-zero
                    
                    voxel = [top_row[i], top_row[i + 1], bottom_row[v], bottom_row[v + 1]]
                    grouped_voxels.append(voxel)
                    i += 1
                    v += 1
                    skip_v = False      # reset skip voxel - used when a voxel type is 0
                else:
                    if not skip_v:      # when voxel type is 0 increment v by 1, helps account for differnece in the top and bottom row array lengths
                        v += 1
                        skip_v = True
                    i += 1

        # filters out voxel that have a type of 0 from the robot structure
        filtered_structure = robot_structure[robot_structure != 0].flatten()

        # convert to NumPy array
        filtered_structure = np.array(filtered_structure)

        # filter so array it is an array of tuples of active voxels
        # the first element in tuple will be voxel_type, second will be the pm_pair indices
        filtered_voxels = [
        (filtered_structure[i], voxel) for i, voxel in enumerate(grouped_voxels) if filtered_structure[i] in {3, 4}
        ]   

        # replace x,y coordinates with their index in the point mass pair array
        filtered_voxels = [
            (voxel_type, [point_index_map[coord] for coord in voxel]) 
            for voxel_type, voxel in filtered_voxels
        ]

        # four corners of the robot
        top_left = sorted_groups_indices[0][0]       # top left - first point in the first row
        top_right = sorted_groups_indices[0][-1]     # top right - last point in the first row
        bottom_left = sorted_groups_indices[-1][0]   # bottom left - first point in last row
        bottom_right = sorted_groups_indices[-1][-1] # bottom right - last point in last row
        robot_corners = [top_left, top_right, bottom_left, bottom_right]

        # store indices of the corners for later use
        self.top_left_corner_index = robot_corners[0]
        self.top_right_corner_index = robot_corners[1]
        self.bottom_left_corner_index = robot_corners[2]
        self.bottom_right_corner_index = robot_corners[3]

        return filtered_voxels, robot_corners

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

        #self.top_left_corner_index = 0
        #self.bottom_right_corner_index = len(self.point_masses) - 1

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
                    pm_pos, self.top_left_corner_index, self.top_right_corner_index, 
                    self.bottom_left_corner_index, self.bottom_right_corner_index))

        return actuator_distances
