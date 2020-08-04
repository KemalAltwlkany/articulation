import numpy as np
from src.PreferenceArticulation.ArticulationExceptions import NotNumpyArray, InvalidComparison

class Solution:
    """
    The solution class is a simple wrapper that contains information about the solution in the SearchSpace, i.e. the
    vector of parameters (x), and the vector of evaluated objective functions (y) for that solution.
    Both attributes x and y are of type np.ndarray.
    """
    eps = 0.00001

    def __init__(self, x=None):
        if x is None:
            x = []
        self.x = x
        self.y = []
        self.val = None

    def set_x(self, x):
        if isinstance(x, np.ndarray):
            self.x = x
        else:
            raise NotNumpyArray('Attempted to assign value to Solution.x which is not of type np.ndarray')

    def set_y(self, y):
        if isinstance(y, np.ndarray):
            self.y = y
        else:
            raise NotNumpyArray('Attempted to assign value to Solution.y which is not of type np.ndarray')

    def set_val(self, val):
        self.val = val

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_val(self):
        return self.val

    def __eq__(self, other):
        """
        Two solution instances are equal if their x-vectors are roughly the same. There is logically
        no need for checking the y vectors as well, since there is a many-to-one mapping.
        "Roughly the same" is defined by class static attribute Solution.eps which defines the relative
        and absolute tolerance allowed between individual coordinates.

        Testing for equality is done using numpys built-in function "isclose" which returns the boolean of the following
        expression:
        absolute(a - b) <= (atol + rtol * absolute(b))
        """
        if isinstance(other, Solution):
            equalities = np.isclose(self.x, other.x, rtol=Solution.eps, atol=Solution.eps)
            return np.all(equalities)
        else:
            raise InvalidComparison('Attempted to compare instance with nonSolution instance.')

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self) -> str:
        s = "x=["
        for xi in self.x:
            s = s + str(xi) + ", "
        s = s + "]\ny=["
        for yi in self.y:
            s = s + str(yi) + ", "
        s = s + "]"
        return s

