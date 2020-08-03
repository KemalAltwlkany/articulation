class ArticulationType(Exception):
    """Exception raised when articulation type is none of the following:
    1.) "apriori"
    2.) "progressive"
    3.) "aposteriori" """

class NotNumpyArray(Exception):
    """Exception raised on attempt to assign anything except a numpy ndarray to solution.x or solution.y"""

class InvalidComparison(Exception):
    """Exception raised on attempt to compare instance of class Solution to any other class."""

class AbstractMethod(Exception):
    """Exception raised on attempt to execute abstract method."""


class ConstraintType(Exception):
    """Exception raised when constraint type is none of the following:
    1.) "EqualityConstraint"
    2.) "GreaterThanConstraint"
    3.) "LessThanConstraint"
    """