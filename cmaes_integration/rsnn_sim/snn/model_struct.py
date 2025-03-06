"""
Module for simulating spiking neural networks (SNNs) with spiky neurons.
"""

import random

# Constants
SPIKE_DECAY = 0.1
MAX_BIAS = 1
MAX_FIRELOG_SIZE = 200


class SpikyNode:
    """
    Class representing a spiky neuron.
    """

    def __init__(self, size):
        self._weights = [
        ]  # a list of weights and a bias (last item in the list)
        self.level = 0.0  # activation level
        self.firelog = []  # tracks whether the neuron fired (1) or not (0)
        self.init(size)

    def init(self, size):
        """Initialize weights and bias."""
        self.firelog.clear()
        if size > 0:
            self._weights = [random.uniform(-1, 1) for _ in range(size)]
            self._weights.append(random.uniform(0, MAX_BIAS))

    def compute(self, inputs):
        """Compute the neuron's output based on inputs."""
        while len(self.firelog) > MAX_FIRELOG_SIZE:
            self.firelog.pop(0)

        # print(f"current level: {self.level}, bias: {self.get_bias()}")
        self.level = max(self.level - SPIKE_DECAY, 0.0)

        if (len(inputs) + 1) != len(self._weights):
            print(
                f"Error: {len(inputs)} inputs vs {len(self._weights)} weights")
            return 0.0

        weighted_sum = sum(inputs[i] * self._weights[i]
                           for i in range(len(inputs)))
        self.level = max(self.level + weighted_sum, 0.0)
        # print(f"new level: {self.level}")

        if self.level >= self.get_bias():
            # print("Fired --> activation level reset to 0.0\n")
            self.level = 0.0
            self.firelog.append(1)
            return 1.0
        # print("\n")
        self.firelog.append(0)
        return 0.0

    def duty_cycle(self):
        """Measures how frequently the neuron fires."""
        if len(self.firelog) == 0:
            return 0.0
        return sum(self.firelog) / len(self.firelog)

    def set_weight(self, idx, val):
        """Sets a weight for a particular node."""
        if 0 <= idx < len(self._weights):
            self._weights[idx] = val
        else:
            print(f"Invalid weight index: {idx}")

    def set_weights(self, input_weights):
        """Allows to set the neuron's weights."""
        if len(input_weights) != len(self._weights):
            print("Weight size mismatch in node")
        else:
            self._weights = input_weights.copy()

    def set_bias(self, val):
        """Sets the neuron's bias."""
        self._weights[-1] = val

    def get_bias(self):
        """Returns the bias from the combined list of weights and bias."""
        return self._weights[-1]

    def print_weights(self):
        """Prints the combined list of weights and bias."""
        print(self._weights)

    @property
    def weights(self):
        """Get the weights of the neuron."""
        return self._weights


class SpikyLayer:
    """
    Collection of multiple neurons (SpikyNodes).
    """

    def __init__(self, num_nodes, num_inputs):
        self.nodes = [SpikyNode(num_inputs) for _ in range(num_nodes)]

    def compute(self, inputs):
        """Feeds input to each node and returns their output."""
        return [node.compute(inputs) for node in self.nodes]

    def set_weights(self, input_weights):
        """Sets weights for all the neurons in the layer."""
        if not self.nodes:
            return
        weights_per_node = len(input_weights) // len(self.nodes)
        for idx, node in enumerate(self.nodes):
            start = idx * weights_per_node
            end = start + weights_per_node
            node.set_weights(input_weights[start:end])

    def duty_cycles(self):
        """Returns the duty cycles for the neurons in the layer."""
        return [node.duty_cycle() for node in self.nodes]


class SpikyNet:
    """
    Combines 2 spiky layers.
    """

    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer = SpikyLayer(hidden_size, input_size)
        self.output_layer = SpikyLayer(output_size, hidden_size)

    def compute(self, inputs):
        """Passes the input through the hidden layer."""
        hidden_output = self.hidden_layer.compute(inputs)
        return self.output_layer.compute(hidden_output)

    def set_weights(self, input_weights):
        """Assigns weights to the hidden and the output layer."""
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


# testing
if __name__ == '__main__':
    print("\n--- Testing SpikyNode ---")
    TEST_NODE = SpikyNode(5)
    print("Initial weights:")
    TEST_NODE.print_weights()

    print("\nSetting weights manually")
    TEST_NODE_WEIGHTS = [0.7, -0.4, 0.9, 0.0, -0.2, 0.8]
    TEST_NODE.set_weights(TEST_NODE_WEIGHTS)
    print("Updated weights:")
    TEST_NODE.print_weights()

    print("\nGetting '1' output for manual input")
    TEST_OUTPUT = TEST_NODE.compute([1, 2, 3, 4, 5])
    print("Output:", TEST_OUTPUT)

    print("\nGetting '0' output for manual input")
    TEST_NODE_WEIGHTS = [0.7, -0.4, -0.9, 0.0, -0.2, 0.8]
    TEST_NODE.set_weights(TEST_NODE_WEIGHTS)
    print("Updated weights:")
    TEST_NODE.print_weights()
    TEST_OUTPUT = TEST_NODE.compute([1, 2, 3, 4, 5])
    print("Output:", TEST_OUTPUT)

    print("\n--- Testing SpikyLayer ---")
    TEST_LAYER = SpikyLayer(3, 4)
    TEST_INPUTS = [1, 2, 3, 4]
    LAYER_OUTPUTS = TEST_LAYER.compute(TEST_INPUTS)
    print("SpikyLayer outputs:", LAYER_OUTPUTS)

    print("\nSetting weights manually")
    TEST_LAYER_WEIGHTS = [0.1 * idx for idx in range(15)]
    TEST_LAYER.set_weights(TEST_LAYER_WEIGHTS)
    print("Updated weights:")
    for node_idx, current_node in enumerate(TEST_LAYER.nodes):
        print(f"Node {node_idx} weights:", current_node.weights)

    print("\n--- Testing SpikyNet ---")
    TEST_NET = SpikyNet(4, 2, 3)
    print("Original structure:")
    TEST_NET.print_structure()
    print("\nTesting computing")
    TEST_NET_OUTPUT = TEST_NET.compute([1, 2, 3, 4])
    print("SpikyNet output:", TEST_NET_OUTPUT)
    print("\nSetting weights manually")
    TEST_NET_WEIGHTS = [
        0.1, 0.2, 0.3, 0.4, 1.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 0.7, 0.0,
        0.0, 0.5, 0.5, 0.5, 0.8
    ]
    TEST_NET.set_weights(TEST_NET_WEIGHTS)
    print("Updated weights:")
    TEST_NET.print_structure()
    print("Getting output for the updated weights")
    TEST_NET_OUTPUT = TEST_NET.compute([1, 2, 3, 4])
    print("SpikyNet output:", TEST_NET_OUTPUT)
