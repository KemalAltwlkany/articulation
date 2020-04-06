from PreferenceArticulation.ArticulationExceptions import AbstractMethod
import math as math


class Constraint:

    def __init__(self, function=None, rel_tol=0.000001, abs_tol=0.000001):
        self.function = function
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol

    def is_satisfied(self, x):
        """
        :param x: A list of floats (decision variables)
        :return: tuple(A,B), where A is of type boolean and indicates whether the constraint is satisfied.
        If it is satisfied, B is 0. If the constraint is not satisfied (i.e. A is False), then
        B holds the evaluation of the constraint function (since the Constraint class is implemented in such a fashion
        that all constraints are compared to number 0, the offset from 0 is indeed the value of the constraint.
        The value of B is therefore used to scale the punishments of the criteria function.
        """
        raise AbstractMethod("Error! Method not overridden in child class.")


class EqualityConstraint(Constraint):

    def __init__(self, function=None, rel_tol=0.000001, abs_tol=0.000001):
        super().__init__(function, rel_tol, abs_tol)

    def is_satisfied(self, x):
        val = self.function(x)
        if math.isclose(val, 0, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            return True, 0  # tuple
        else:
            return False, math.fabs(val)
        # returns math.fabs(val) because it is assumed that all objectives are to be minimized
        # and the penalties are therefore implemented as objective = objective + M*val
        # therefore, val must be positive, otherwise a counter-effect is created, and the algorithm would
        # prefer the unacceptable solution


class GreaterThanConstraint(Constraint):

    def __init__(self, function=None, rel_tol=0.000001, abs_tol=0.000001):
        super().__init__(function, rel_tol, abs_tol)

    def is_satisfied(self, x):
        val = self.function(x)
        if val > 0:
            return True, 0
        else:
            return False, math.fabs(val)


class LessThanConstraint(Constraint):

    def __init__(self, function=None, rel_tol=0.000001, abs_tol=0.000001):
        super().__init__(function, rel_tol, abs_tol)

    def is_satisfied(self, x):
        val = self.function(x)
        if val < 0:
            return True, 0
        else:
            return False, math.fabs(val)
        # returns math.fabs(val) because it is assumed that all objectives are to be minimized
        # and the penalties are therefore implemented as objective = objective + M*val
        # therefore, val must be positive, otherwise a counter-effect is created, and the algorithm would
        # prefer the unacceptable solution


# Warning - not inherited from class Constraint.
class BoundaryConstraint:
    def __init__(self, boundaries=None, rel_tol=0.000001, abs_tol=0.000001):
        """
        To avoid the effect of medial and extremal parameters, a boundary check is performed to verify whether
        all decision variables are within scope. If not, the criteria is punished thereby making searches in
        out of boundary regions less appealing.
        This however, does not necessarily make it impossible to implement functions that return the search back to the
        search space by setting a wandering decision variable equal to its medial or extremal value (or whats more, to
        any random value within scope).
        :param boundaries: A list of length n, containing tuples. A tuple at the i-th list index represents the
        min and max values respectively of the i-th decision variable.
        e.g.
        x1 € [-5, 5], x €[17, 19]
        boundaries = [(-5, 5), (17,19)]
        """
        self.boundaries = boundaries
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol

    def is_satisfied(self, x):
        val = 0
        satisfied = True
        for xi, bound in zip(x, self.boundaries):
            if xi - bound[0] < 0 or xi - bound[1] > 0:
                val = val + 1
                satisfied = False
        return satisfied, val


def TNK_constraint_1(x):
    if math.isclose(x[0], 0):
        return math.pow(x[1], 2) - 1.1 # atan not defined for x[0] = 0. Assume cos(atan(inf))=1
    return math.pow(x[0], 2) + math.pow(x[1], 2) - 1 - 0.1*math.cos(16*math.atan(x[1]/x[0]))

def TNK_constraint_2(x):
    return math.pow(x[0] - 0.5, 2) + math.pow(x[1] - 0.5, 2) - 0.5

def OSY_constraint_1(x):
    return x[0] + x[1] - 2

def OSY_constraint_2(x):
    return 6 - x[0] - x[1]

def OSY_constraint_3(x):
    return 2 + x[0] - x[1]

def OSY_constraint_4(x):
    return 2 - x[0] + 3*x[1]

def OSY_constraint_5(x):
    return 4 - math.pow(x[2] - 3, 2) - x[3]

def OSY_constraint_6(x):
    return math.pow(x[4] - 3, 2) + x[5] - 4
