import numpy as np

# Constants
DEFAULT_SIZE = 50


class RingBuffer:
    """Class representing ring buffer"""

    def __init__(self, size=DEFAULT_SIZE):
        self.size = size
        self.data = np.zeros(size, dtype=float)  # Initialize with zeros
        self.position = 0
        self.count = 0

    def add(self, x):
        """Add element to buffer"""
        self.data[self.position] = x
        self.position = (self.position + 1) % self.size
        self.count = min(self.count + 1, self.size)
    
    def get(self):
        """Returns list of elements in correct order"""
        if self.count < self.size:
            return self.data[:self.count]
        return np.roll(self.data, -self.position)

    def is_full(self):
        """Check if buffer is full"""
        return self.count == self.size

    def clear(self):
        """Clear the buffer"""
        self.data = np.zeros(self.size, dtype=float)
        self.position = 0
        self.count = 0

# Testing
if __name__ == '__main__':
    rb = RingBuffer(10)
    rb.add(4)
    rb.add(5)
    rb.add(10)
    rb.add(7)
    print(rb.__class__, rb.get())

    data = [1, 11, 6, 8, 9, 3, 12, 2]
    for value in data[:6]:
        rb.add(value)
    print(rb.__class__, rb.get())

    print('')