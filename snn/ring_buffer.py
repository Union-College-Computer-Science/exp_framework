"""
Module for ring buffer implementation.
"""

import numpy as np


class RingBuffer:
    """ Implements a partially-full buffer. """

    def __init__(self, buffsize):
        self.buffsize = buffsize
        self.data = np.empty(
            buffsize,
            dtype=object)  # Numpy array of 'buffsize' with None values
        self.currpos = 0  # Position where the next element should be added
        self.is_full = False  # Flag to indicate if the buffer is full

    def add(self, value):
        """ Add an element at the end of the buffer. """
        self.data[self.currpos] = value
        self.currpos = (self.currpos + 1) % self.buffsize
        if self.currpos == 0:
            self.is_full = True

    def get(self):
        """ Return a list of elements from the oldest to the newest without None values """
        if self.is_full:
            return np.concatenate(
                (self.data[self.currpos:], self.data[:self.currpos]))
        return self.data[:self.currpos]


# testing
if __name__ == '__main__':

    # Creating ring buffer
    x = RingBuffer(10)

    # Adding first 4 elements
    x.add(5)
    x.add(10)
    x.add(4)
    x.add(7)

    # Displaying class info and buffer data
    print(x.__class__, x.get())

    # Creating fictitious sampling data list
    data = [1, 11, 6, 8, 9, 3, 12, 2]

    # Adding elements until buffer is full
    for value in data[:6]:
        x.add(value)

    # Displaying class info and buffer data
    print(x.__class__, x.get())

    # Adding data simulating a data acquisition scenario
    print('')
    print(f'Mean value = {np.mean(x.get()):0.1f}   |  ', x.get())
    for value in data[6:]:
        print(f"Adding {value}")
        x.add(value)
        print(f'Mean value = {np.mean(x.get()):0.1f}   |  ', x.get())
