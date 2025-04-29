"""
Module for simulating spiking neural networks (SNNs) with spiky neurons.

Authors: Abhay Kashyap, Atharv Tekurkar
"""

import numpy as np
from snn.ring_buffer import RingBuffer

# Constants
SPIKE_DECAY = 0.01
MAX_BIAS = 1
MAX_FIRELOG_SIZE = 10

class SpikyNode:
    """
    Class representing a spiky neuron.
    """

    def __init__(self, size):
        """
        Initializes a spike neuron.

        Parameters:
            size (int): Number of weights plus the bias.
        """
        # a list of weights and a bias (last item in the list)

        self._weights = np.random.uniform(-0.3, 0.3, (size + 1))
        self.level = 0  # activation level
        self.buffer = RingBuffer(
            MAX_FIRELOG_SIZE)  # tracks whether the neuron fired or not
        self.levels_log = []
        self.fire_log = []
        self.duty_cycle_log = []

    def compute(self, inputs):
        """
        Compute the neuron's output based on inputs.
        
        Parameters:
            inputs (list): Inputs into this node.

        Returns:
            tuple: (1.0 if the neuron fired, 0.0 otherwise,
                    the neuron's current level).
        """

        self.level *= (1 - SPIKE_DECAY)

        if (len(inputs) + 1) != len(self._weights):
            print(f"Error: {len(inputs)} inputs vs {len(self._weights)} \
                  weights; weights: {self._weights}")
            return 0.0, self.level

        weighted_sum = sum(inputs[i] * self._weights[i]
                           for i in range(len(inputs)))
        
        self.level += weighted_sum

        self.levels_log.append(self.level)

        if self.level >= self.get_bias(): # Neuron fires

            self.level = 0
            self.fire_log.append(1)
            self.buffer.add(1)
            self.duty_cycle_log.append(self.duty_cycle())
            return 1.0, self.level
        else:                             # Neuron doesn't fire
            
            self.fire_log.append(0)
            self.buffer.add(0)
            self.duty_cycle_log.append(self.duty_cycle())
            return 0.0, self.level

    def duty_cycle(self):
        """
        Measures how frequently the neuron fires.

        Parameters:
            window (int): How far back to compute the duty cycle.

        Returns:
            float: What pecent of the last `MAX_FIRELOG_SIZE` timesteps the neuron fired, in decimal form.
        """
        if self.buffer.length() == 0:
            return 0.0

        recent_fires = self.buffer.get()
    
        return recent_fires.count(1)/MAX_FIRELOG_SIZE


    def set_weights(self, input_weights):
        """
        Sets the neuron's weights.
        
        Parameters:
            input_weights (list): Incoming weights into the neuron.
        """
        if len(input_weights) != len(self._weights):
            print("Weight size mismatch in node")
        else:
            self._weights = input_weights.copy()
            self._weights[:-1] = list(map(lambda x: abs(x), self._weights[:-1]))
            # self._weights = input_weights.copy()

    def set_bias(self, val):
        """Sets the neuron's bias.
        
        Parameters:
            val (float): Value for neuron bias.
        """
        self._weights[-1] = val

    def get_bias(self):
        """
        Returns the bias from the combined list of weights and bias.

        Returns:
            float: The neuron's bias.
        """
        return self._weights[-1]

    def print_weights(self):
        """
        Prints the combined list of weights and bias.
        
        Returns:
            list: A list of weights, with the last entry being the neuron's bias.
        """
        print(self._weights)

    def get_levels_log(self):
        """
        Return the list of the neuron's recent activation levels.
        
        Returns:
            list: A list of all the neuron's levels.
        """
        return self.levels_log
    
    def get_fire_log(self):
        """
        Return the list of the neuron's firelog.
        
        Returns:
            list: A list representing the neuron's firelog.
        """
        return self.fire_log
    
    def get_duty_cycle_log(self):
        """
        Return the neuron's duty cycle for each timestep.
        
        Returns:
            list: A list of all the neuron's duty cycles for each time step..
        """
        return self.duty_cycle_log

    @property
    def weights(self):
        """Get the weights of the neuron."""
        return self._weights


class SpikyLayer:
    """
    Collection of multiple neurons (SpikyNodes).
    """

    def __init__(self, num_nodes, num_inputs):
        """
        Initializes a SpikyLayer.

        Parameters:
            num_nodes (int): Number of neurons in the layer.
            num_inputs (int): Number of inputs into each neuron the layer.
        """

        self.nodes = [SpikyNode(num_inputs, spike_decay)
                      for _ in range(num_nodes)]


    def compute(self, inputs):
        """
        Feeds input to each node and returns their output.
        
        Parameters:
            inputs (list): A list of inputs into this layer.

        Returns:
            tuple: (a list of all neuron outputs, a list of all neuron levels)
        """

        outputs = []
        levels = []

        for node in self.nodes:
            output, level = node.compute(inputs)
            outputs.append(output)
            levels.append(level)

        return outputs, levels

    def set_weights(self, input_weights):
        """
        Sets weights for all the neurons in the layer.
        
        Parameters:
            input_weights (list): List of weights for all neurons in the layer.
        """
        if not self.nodes:
            return
        weights_per_node = len(input_weights) // len(self.nodes)
        for idx, node in enumerate(self.nodes):
            start = idx * weights_per_node
            end = start + weights_per_node
            node.set_weights(input_weights[start:end])

    def duty_cycles(self):
        """
        Returns the duty cycles for the neurons in the layer.
        
        Parameters:
            window (int): How many previous time steps to compute the duty cycle.

        Returns:
            list: The duty cycle for each neuron in the layer.
        """
        return [node.duty_cycle() for node in self.nodes]


class SpikyNet:
    """
    Combines 2 spiky layers, a hidden layer and an output layer.
    """

    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Initializes network.
        
        Parameters:
            input_size (int): Number of inputs into the network.
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of outputs.
        """

        self.hidden_layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layer = SpikyLayer(int(hidden_size), prev_size)
            self.hidden_layers.append(layer)
            prev_size = hidden_size

        self.output_layer = SpikyLayer(output_size, prev_size)

    def compute(self, inputs):
        """
        Passes the input through all hidden layers, then output layer.

        Parameters:
            inputs (list): Inputs to the network.
        """

        hidden_output, hidden_levels = self.hidden_layer.compute(inputs)
        output, levels = self.output_layer.compute(hidden_output)

        return output, levels

    def set_weights(self, input_weights):
        """
        Assigns weights to all hidden layers and the output layer.
        
        Parameters:
            input_weights: A dictionary with two keys: 'hidden_layer', mapping to a list
                           of all that layer's weights and biases, and a key 'output_layer',
                           mapping to a list of all that layers weights and biases.
        """
        self.hidden_layer.set_weights(input_weights['hidden_layer'])
        self.output_layer.set_weights(input_weights['output_layer'])

    def print_structure(self):
        """Displays the network weights."""
        print("Hidden Layer:")
        for node_index, hidden_node in enumerate(self.hidden_layer.nodes):
            print(f"Node {node_index}: ", end="")
            hidden_node.print_weights()
        print("\nOutput Layer:")
        for node_index, output_node in enumerate(self.output_layer.nodes):
            print(f"Node {node_index}: ", end="")
            output_node.print_weights()
        print("\n")
