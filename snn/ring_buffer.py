import numpy as np

class RingBuffer:
    """ Implements a partially-full buffer. """
    def __init__(self, buffsize):
        self.buffsize = buffsize
        self.data = np.empty(buffsize, dtype=object)    # numpy array of size 'buffsize' with None values
        self.currpos = 0    # position where the next element should be added
        self.count = 0  # number of elements in the buffer

    class __Full:
        """ Implements a full buffer """
        def add(self, x):
            """ Add an element overwriting the oldest one. """
            self.data[self.currpos] = x
            self.currpos = (self.currpos+1) % self.buffsize

        def get(self):
            """ Return list of elements from the oldest to the newest. """
            return np.concatenate((self.data[self.currpos:], self.data[:self.currpos]))

    def add(self,x):
        """ Add an element at the end of the buffer. """
        self.data[self.currpos] = x
        self.currpos = (self.currpos + 1) % self.buffsize
        self.count += 1
        
        if self.count == self.buffsize:
            # Change self's class from not-yet-full to full
            self.__class__ = self.__Full

    def get(self):
        """ Return a list of elements from the oldest to the newest without None values """
        return np.array([val for val in self.data[:self.count] if val is not None])


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
    print('Mean value = {:0.1f}   |  '.format(np.mean(x.get())), x.get())
    for value in data[6:]:
        print("Adding {}".format(value))
        x.add(value)
        print('Mean value = {:0.1f}   |  '.format(np.mean(x.get())), x.get())