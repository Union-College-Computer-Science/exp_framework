"""module for ring buffer data structure"""
import numpy as np

#constants
DEFAULT_SIZE = 50


class RingBuffer:
    """class representing ring buffer"""

    def __init__(self, size=DEFAULT_SIZE):
        self.size = size
        self.data = None  # Will be initialized on first add
        self.position = 0
        self.count = 0

    def add(self, x):
        """add element to buffer, overwriting old elements when full"""
        x = np.array(x)
        if self.data is None:
            self.data = np.zeros((self.size,) + x.shape, dtype=x.dtype)
        
        self.data[self.position] = x
        self.position = (self.position + 1) % self.size
        self.count = min(self.count + 1, self.size)
    
    def get(self):
        """returns list of elements in correct order"""
        if self.count == 0:
            return np.array([])
        if self.count < self.size:
            return self.data[:self.count]
        return np.concatenate([self.data[self.position:], self.data[:self.position]])

    def is_full(self):
        """returns whether the buffer is full"""
        return self.count >= self.size

# Testing
if __name__ == '__main__':
    rb = RingBuffer(10)
    rb.add(4)
    rb.add(5)
    rb.add(10)
    rb.add(7)
    print("Initial state:", rb.get())

    data = [1, 11, 6, 8, 9, 3, 12, 2]
    for value in data[:6]:
        rb.add(value)
    print("After adding more data:", rb.get())