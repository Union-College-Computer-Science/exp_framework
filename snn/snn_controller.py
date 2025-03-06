"""
Module for running SNN outputs with proper input/output handling.
"""

import json
import os
import numpy as np
from snn.model_struct import SpikyNet


# Constants for SNN configuration
MIN_LENGTH = 0.6  # Minimum actuator length
MAX_LENGTH = 1.6  # Maximum actuator length
_current_file = os.path.abspath(__file__)
_project_root = os.path.dirname(os.path.dirname(_current_file))
ROBOT_DATA_PATH = os.path.join(_project_root, "morpho_demo", "world_data", "bestbot.json")


class SNNController:
    """Class to handle SNN input/output processing."""
    def __init__(self, inp_size, hidden_size, output_size, robot_config=ROBOT_DATA_PATH):
        """Initialize with None - will set sizes after loading robot data."""
        self.snns = []
        self.num_snn = 0  # Number of spiking neural networks (actuators)
        self.inp_size = inp_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self._load_robot_config(robot_config)

    def _load_robot_config(self, robot_path):
        """
        Load robot configuration from JSON file and initialize SNN.
        
        Args:
            robot_path (str): Path to robot JSON configuration file
            
        Returns:
            tuple: (num_actuators, input_size) - Network dimensions
        """
        if not os.path.exists(robot_path):
            raise FileNotFoundError(f"Robot configuration file not found: {robot_path}")
        with open(robot_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Extract robot data
        robot_key = list(data["objects"].keys())[0]
        robot_data = data["objects"][robot_key]
        # Count actuators (types 3 and 4)
        self.num_snn = sum(1 for t in robot_data["types"] if t in [3, 4])
        # Initialize SNN with proper dimensions
        self.snns = [SpikyNet(
            input_size=self.inp_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size
        ) for _ in range(self.num_snn)]

    def set_snn_weights(self, cmaes_out):
        """
        Retrieve the flat CMA-ES output and 
        reshape it into a structured format for the SNN's `set_weights()`.

        Returns:
            snn_parameters: A dictionary containing the weights and biases for each SNN.
                        - dict with two elements : 'hidden_layer' and 'output_layer'
                            'hidden_layer' - weights and biases for all nodes in the hidden layer
                            'output_layer' - weights and biases for all nodes in the output layer
                        
                            
        Raises:
            ValueError: If the length of the CMA-ES output does not match the expected size.
        """

        params_per_hidden_layer = (self.inp_size + 1) * self.hidden_size
        params_per_output_layer = (self.hidden_size + 1) * self.output_size
        params_per_snn = params_per_hidden_layer + params_per_output_layer

        flat_vector = np.array(cmaes_out)  # np.array(pipeline.get_cmaes_out())

        if flat_vector.size != (self.num_snn * params_per_snn):
            raise ValueError(f"Expected CMA-ES output vector of size \
                             {(self.num_snn * params_per_snn)}, got {flat_vector.size}.")

        # Reshape the flat vector to a 2D array: each row corresponds to one SNN.
        reshaped = flat_vector.reshape((self.num_snn, params_per_snn))

        # For each SNN, split the parameters into weights and biases.
        snn_parameters = {}
        for snn_idx, params_per_snn in enumerate(reshaped):
            hidden_params = params_per_snn[:params_per_hidden_layer]
            output_params = params_per_snn[params_per_hidden_layer:]
            snn_parameters[snn_idx] = {
                'hidden_layer': hidden_params,
                'output_layer': output_params
            }

        for snn_id, params in snn_parameters.items():
            self.snns[snn_id].set_weights(params)

    def _get_output_state(self, inputs):
        """
        Run SNN with inter-actuator distances as input over multiple timesteps.
        
        Args:
            inputs (list): inter-actuator distances
            
        Returns:
            dict: Contains 'continuous_actions' and 'duty_cycles'
        """
        # inputs = morpho.get_inputs()
        outputs = {}
        for snn_id, snn in enumerate(self.snns):
            snn.compute(inputs[snn_id])
            duty_cycle = snn.output_layer.duty_cycles()
            scale_factor = MAX_LENGTH - MIN_LENGTH
            scaled_actions = [(dc * scale_factor) + MIN_LENGTH for dc in duty_cycle]
            outputs[snn_id] = {
                "target_length": scaled_actions,
                "duty_cycle": duty_cycle
                }
        return outputs

    def get_lengths(self, inputs):
        """
        Returns a list of target lengths (action array)
        """
        out = self._get_output_state(inputs)
        lengths = []
        for _, item in out.items():
            lengths.append(item['target_length'])
        return lengths


def calc_param_num(inp_size, hidden_size, out_size):
    """
    Returns the total number of parameters per snn
    """
    params_per_hidden_layer = (inp_size + 1) * hidden_size
    params_per_output_layer = (hidden_size + 1) * out_size
    params_per_snn = params_per_hidden_layer + params_per_output_layer
    return params_per_snn

def main():
    """Main function to demonstrate SNN output generation."""
    runner = SNNController(inp_size=7, hidden_size=3, output_size=1)
    inp_size = runner.inp_size
    num_snn = runner.num_snn
    try:
        # Load robot configuration and initialize SNN
        print(f"Initialized SNN with {inp_size} inputs")
        # Generate random weights for testing
        print(f"params per snn: {calc_param_num(inp_size, 3, 1)}")
        num_weights = calc_param_num(inp_size, runner.hidden_size, runner.output_size) * num_snn
        test_weights = np.random.rand(num_weights)
        inputs = [np.random.random(inp_size) for _ in range(num_snn)]
        print(f"inputs: {inputs}")
        runner.set_snn_weights(test_weights)
        # Generate outputs
        print("\nRunning get_output_state for 100 steps...")
        output_states = runner.get_lengths(inputs)
        print(output_states)
        # runner.save_output_state(output_states, "snn_outputs.json")
        # print("\nSaved outputs to snn_outputs.json")
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    main()
