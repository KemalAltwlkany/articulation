import math as math


class Solution:
    """
    The solution class is a simple wrapper that contains information about the solution in the SearchSpace, i.e. the
    vector of parameters (x), and the vector of evaluated objective functions (y) for that solution.
    """
    eps = 0.0001

    def __init__(self, x):
        self.x = x
        self.y = []

    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y = y

    def __eq__(self, other):
        """
        Two solution instances are equal if their x-vectors are roughly the same. There is logically
        no need for checking the y vectors as well, since there is a many-to-one mapping.
        "Roughly the same" is defined by class static attribute Solution.eps which defines the relative
        and absolute tolerance allowed between individual coordinates.
        """
        if isinstance(other, Solution):
            for i, j in zip(self.x, other.x):
                if math.isclose(i, j, rel_tol=Solution.eps, abs_tol=Solution.eps):
                    continue
                else:
                    return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)



    def __str__(self) -> str:
        s = "x=["
        # s = s + str(len(self.x)) + "\n"
        # s = s + str(len(self.y)) + "\n"
        for xi in self.x:
            s = s + str(xi) + ", "
        s = s + "]\ny=["
        for yi in self.y:
            s = s + str(yi) + ", "
        s = s + "]"
        return s

    # old version of str(Solution)
    # def __str__(self):
    #     str_ = "\n------ Class solution -------\n"
    #     str_ = str_ + "Decision variables: ["
    #     for ind, xi in enumerate(self.x):
    #         if ind is not len(self.x)-1:
    #             str_ = str_ + str(xi) + ", "
    #         else:
    #             str_ = str_ + str(xi) + "]"
    #
    #     str_ = str_ + "\nObjectives values: ["
    #     for ind, yi in enumerate(self.y):
    #         if ind is not len(self.y)-1:
    #             str_ = str_ + "f" + str(ind+1) + "=" + str(yi) + ", "
    #         else:
    #             str_ = str_ + "f" + str(ind+1) + "=" + str(yi) + "]"
    #     str_ = str_ + "\n------ ------------- -------"
    #     return str_




