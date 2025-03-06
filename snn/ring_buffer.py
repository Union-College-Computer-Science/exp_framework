"""module for ring buffer data structure"""
import numpy as np

#constants
DEFAULT_SIZE = 50


class RingBuffer:
    """class representing ring buffer"""

    def __init__(self, size=DEFAULT_SIZE):
        self.size = size
        self.data = np.array

    class __FullBuffer:
        def add(self, x):
            """add element, overwrite old element"""
            self.data[self.position] = x
            self.position = (self.position +1) % self.size
        
        def get(self):
            """returns list of elements in correct order"""
            return self.data[self.position:]+self.data[:self.position]

    def add(self, x):
        """add elem to end of buffer"""
        np.append(self.data, x)
        if np.size(self.data) == self.size:
            self.position = 0
            self.__class__ = self.__FullBuffer

    def get(self):
        return self.data

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
    print (rb.__class__, rb.get())

    print ('')