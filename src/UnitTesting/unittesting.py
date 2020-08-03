import unittest
import numpy as np
from src.PreferenceArticulation.Solution import Solution

class TestSolution(unittest.TestCase):
    def setUp(self):
        Solution.eps = 0.000001
        self.sol1 = Solution()
        self.sol2 = Solution()


    def tearDown(self):
        self.sol1 = None
        self.sol2 = None

    def test_success_assignation_x(self):
        arr_1 = np.array([1, 2, 3])
        self.sol1.set_x(arr_1)
        self.assertTrue(np.all(np.isclose(arr_1, self.sol1.x, rtol=Solution.eps, atol=Solution.eps)))

    @unittest.expectedFailure
    def test_fail_assignation_x(self):
        k = [1, 2, 3]
        self.sol1.set_x(k)

    def test_success_assignation_y(self):
        arr_1 = np.array([1, 2, 3])
        self.sol1.set_y(arr_1)
        self.assertTrue(np.all(np.isclose(arr_1, self.sol1.y, rtol=Solution.eps, atol=Solution.eps)))

    @unittest.expectedFailure
    def test_fail_assignation_y(self):
        k = [1, 2, 3]
        self.sol1.set_y(k)


    def test_success_equality(self):
        self.sol1 = Solution(np.array([1, 2, 3, 4, 5]))
        self.sol2 = Solution(np.array([1.0000001, 2.00000001, 3.0000001, 4.0000001, 5.000001]))
        self.assertTrue(self.sol1 == self.sol2)

    @unittest.expectedFailure
    def test_fail_equality(self):
        self.sol1 = Solution(np.array([1, 2, 3, 4, 5]))
        self.sol2 = Solution(np.array([1.0000001, 2.00002, 3.0000001, 4.0000001, 5.000001]))
        self.assertTrue(self.sol1 == self.sol2)

    def test_success_nequality(self):
        self.sol1 = Solution(np.array([1, 2, 3, 4, 5]))
        self.sol2 = Solution(np.array([1.0000001, 2.00002, 3.0000001, 4.0000001, 5.000001]))
        self.assertTrue(self.sol1 != self.sol2)

