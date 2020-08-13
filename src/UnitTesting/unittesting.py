import unittest
import numpy as np
import random as random
import pickle as pickle

from src.PreferenceArticulation.Solution import Solution
from src.PreferenceArticulation.BenchmarkObjectives import *
from src.ProgressiveArticulation.rule_based_tree import RuleBasedTree


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


class TabuTests(unittest.TestCase):
    def setUp(self):
        Solution.eps = 0.00001

    def test_solution_comparisons(self):
        # this method tests whether a solution is in a list (tabu list)
        x1, x2, x3 = np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0]), np.array([1.00001, 2.0, 3.0])
        tabu_list = [Solution(x1), Solution(x2), Solution(x3)]
        sol = Solution(np.array([1.000000001, 2.00000001, 3.0]))
        self.assertTrue(sol in tabu_list)


class ProblemTests(unittest.TestCase):

    def test_FON_evaluation(self):
        # method tests whether the FON criteria is evaluated correctly using numpy vectorization
        # comparison is drawn using slowly, step-by-step calculations
        evaluation = MOO_Problem.FON

        def compute_FON_manually(x):
            n = len(x)
            sum1 = 0
            sum2 = 0
            for k in range(n):
                sum1 = sum1 - math.pow(x[i] - 1. / math.sqrt(n), 2)
                sum2 = sum2 - math.pow(x[i] + 1. / math.sqrt(n), 2)
            return np.array([1 - math.exp(sum1), 1 - math.exp(sum2)])

        # perform the test 10 times for dimensions 1-10, on random variables
        np.random.seed(0)
        random.seed(0)
        for i in range(1, 10):
            for j in range(10):
                x = 8 * (0.5 - random.random()) * np.random.rand(i)
                manual, numpyist = compute_FON_manually(x), evaluation(x)
                self.assertAlmostEqual(manual[0], numpyist[0])
                self.assertAlmostEqual(manual[1], numpyist[1])

    def test_OSY_evaluation(self):
        evaluation = MOO_Problem.OSY

        def manual_f1(a):
            return -25 * (a[0] - 2) ** 2 - (a[1] - 2) ** 2 - (a[2] - 1) ** 2 - (a[3] - 4) ** 2 - (a[4] - 1) ** 2

        def manual_f2(a):
            return a[0] ** 2 + a[1] ** 2 + a[2] ** 2 + a[3] ** 2 + a[4] ** 2 + a[5] ** 2

        for i in range(1000):
            np.random.seed(i)
            x = np.random.rand(6)
            self.assertAlmostEqual(evaluation(x)[0], manual_f1(x))
            self.assertAlmostEqual(evaluation(x)[1], manual_f2(x))


class ConstraintTests(unittest.TestCase):
    def test_constraint_BK1(self):
        constraint = MOO_Constraints.BK1_constraint
        x = np.array([-3, -3])
        self.assertEqual(constraint(x), 0)

        x = np.array([-5.0000001, 9])
        self.assertEqual(constraint(x), 1)

        x = np.array([9.999999, -5.000001])
        self.assertEqual(constraint(x), 1)

        x = np.array([-4.999999, 9.999999])
        self.assertEqual(constraint(x), 0)

        x = np.array([0, 0])
        self.assertEqual(constraint(x), 0)

    def test_constraint_IM1(self):
        constraint = MOO_Constraints.IM1_constraint
        x = np.array([1.00001, 1.00001])
        self.assertEqual(constraint(x), 0)

        x = np.array([0.999999, 1.000001])
        self.assertEqual(constraint(x), 1)

        x = np.array([4.0000001, 1.5])
        self.assertEqual(constraint(x), 1)

        x = np.array([3.5, 0.999999])
        self.assertEqual(constraint(x), 1)

        x = np.array([1.00001, 2.000001])
        self.assertEqual(constraint(x), 1)

        x = np.array([0.99999, 0.99999])
        self.assertEqual(constraint(x), 2)

        x = np.array([4.000001, 2.000001])
        self.assertEqual(constraint(x), 2)

        x = np.array([2.2, 1.7])
        self.assertEqual(constraint(x), 0)

        x = np.array([17, -5])
        self.assertEqual(constraint(x), 2)

    def test_constraint_FON(self):
        constraint = MOO_Constraints.FON_constraint
        x = np.array([-4, 0.35, 4.0000001])
        self.assertEqual(constraint(x), 1)

        x = np.array([-3.99999, 3.99999, 3.999998, -3.99998])
        self.assertEqual(constraint(x), 0)

        x = np.array([-4.000001, 4.0000001, 17, -15.4])
        self.assertEqual(constraint(x), 4)

        x = np.array([3.5, 0.01, 2, 1, 3, 3.999, -3.8, -0.1])
        self.assertEqual(constraint(x), 0)

    def test_constraint_TNK_1(self):
        constraint = MOO_Constraints.TNK_constraint_1

        def manual_computation(a):
            sum_ = 0
            if a[0] < 0 or a[0] > math.pi:
                sum_ += 1
            if a[1] < 0 or a[1] > math.pi:
                sum_ += 1
            return sum_

        for i in range(500):
            x = np.random.uniform(low=-5, high=5, size=2)
            self.assertEqual(constraint(x), manual_computation(x))

    def test_constraint_TNK_2(self):
        constraint = MOO_Constraints.TNK_constraint_2
        
        def manual_computation(a):
            if math.isclose(a[0], 0, rel_tol=0.000000001, abs_tol=0.0000000001):
                left_side = a[1]**2
                right_side = 1.1
            else:
                left_side = a[0]**2 + a[1]**2
                right_side = 1 + 0.1*math.cos(16*math.atan(a[1]/a[0]))
            return int(left_side >= right_side)
        
        for i in range(500):
            x = np.random.uniform(low=-5, high=5, size=2)
            self.assertEqual(constraint(x), manual_computation(x))

    def test_constraint_TNK_3(self):
        constraint = MOO_Constraints.TNK_constraint_3

        def manual_computation(a):
            left_side = (a[0] - 0.5)**2 + (a[1] - 0.5)**2
            right_side = 0.5
            return int(left_side <= right_side)

        for i in range(100):
            x = np.random.uniform(low=-5, high=5, size=2)
            self.assertEqual(constraint(x), manual_computation(x))

        x = np.array([1.1727, 0.435])
        print(constraint(x))
        self.assertEqual(constraint(x), manual_computation(x))

