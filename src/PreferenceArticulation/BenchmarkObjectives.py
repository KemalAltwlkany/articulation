import math as math
import numpy as np

class MOO_Problem:
    """
    Class is never to be instantiated in the first place, its just a wrapper to keep all the benchmark functions used
    in one nice class, which should be used as static functions.
    """

    @staticmethod
    def BK1(x):
        """
        The BK1 test problem.
        In the work of S. Huband, this test problem is labeled "BK1"
        From T.T.Binh, U. Korn - "An evolution strategy for the multiobjective optimization"; page 4/6
        A simple bi-objective problem
            f1(x1, x2) = x1**2 + x2**2
            f2(x1, x2) = (x1-5)**2 + (x2-5)**2
        Region is defined as x1 € [-5, 10] and x2 € [-5, 10]
        Characteristics:
        f1: Separable, Unimodal
        f2: Separable, Unimodal
        Pareto front convex
        The Pareto front is defined for x1 € [0, 5] and x2 € [0,5].
        This is logical, because the first function is optimized for (0,0) and the second for (5, 5). Any inbetween solutions
        due to the linear derivative of the 2 objectives is a trade-off.

        R1 - y, R2 - no, R3 - no, R4 - no, R5 - no, R6 - no, R7 - no
        :param x: a list of floats containing the solution's vector of decision variables. Basically, x = Sol.x
        :return: a 2 element list, containing the evaluated objectives; f1, f2 respectively.
        """
        f1 = x[0]**2 + x[1]**2
        f2 = (x[0] - 5)**2 + (x[1] - 5)**2
        return np.array([f1, f2])

    @staticmethod
    def IM1(x):
        """
        The IM1 Test problem.
        In the work of S. Huband, this test problem is labeled "IM1"
        From: H. Ishibuchi,T. Murata;
        "A multi-objective genetic local search algorithm and its application to flowshop scheduling"
        Test problem 2:
            minimize: f1(x1, x2) 2*sqrt(x1)
                f2(x1, x2) = x1*(1-x2) + 5
                x1 € [1, 4], x2 € [1, 2]
        Interesting problem because of a nonconvex fitness space. Weighted algorithms perform poorly on nonconvex spaces.
        f1 - unimodal
        f2 - unimodal
        R1 - no, R2 - yes, R3 - no, R4 - no, R5 - yes, R6 - yes, R7 - yes
        Fitness space is CONCAVE.

        Pareto optimal front is obtain for x2=2.
        Cited from:
        M. Tadahiko, H. Ishibuchi - MOGA: Multi-Objective Genetic Algorithms
        :param x: a list of floats containing the solution's vector of decision variables. Basically, x = Sol.x
        :return: a 2 element list, containing the evaluated objectives; f1, f2 respectively.
        """
        f1 = 2*math.sqrt(x[0])
        f2 = x[0]*(1-x[1]) + 5
        return [f1, f2]

    @staticmethod
    def SCH1(x):
        """
        SCH1 - Schaffers test function, from Schaffer 1984, I cited it from:
        "K. Deb, L. Thiele, M. Laumanns, E. Zitzler - Scalable test problems for evolutionary multi-objective optimization"
        The Pareto optimal front can be obtained for any:
        x € [0, 2] (Pareto optimal set).
        :return:
        """
        return [x[0]**2, (x[0] - 2)**2]

    @staticmethod
    def FON(x):
        """
        This problem is obtained from the paper:
        "K. Deb, L. Thiele, M. Laumanns, E. Zitzler - Scalable test problems for evolutionary multi-objective optimization"
        it can be found under equation 1, on page 5/28.
        It is a search-space-wise scalable problem, in n-dimensions, but with only 2 objectives.
        The Pareto optimal set contains all solutions defined by:
        xi € [-1/sqrt(n), 1/sqrt(n)], where n is the number of decision variables.
        Note: all xi are equal in the Pareto front, that is x1 € [-1/sqrt(n), 1/sqrt(n)] and
            all other xi = x1, for i=2,...,n
        The search space is limited to -4<= x <= 4
        :return:
        """
        n = len(x)
        sum1 = 0
        sum2 = 0
        for i in range(n):
            sum1 = sum1 - math.pow(x[i] - 1./math.sqrt(n), 2)
            sum2 = sum2 - math.pow(x[i] + 1./math.sqrt(n), 2)
        return [1 - math.exp(sum1), 1 - math.exp(sum2)]

    @staticmethod
    def TNK(x):
        """
        I've found about the paper:
        "K. Deb, A. Pratap, T. Meyarivan - Constrained Test Problems for Multi-objective Evolutionary Optimization"
        It can be found on pages 3 and 4. The problem definition however contains a mistake in this paper.
        The problem can be originally found in the work of:
        "Tanaka - GA-based decision support system for multicriteria optimization"
        where it is introduced and written correctly. The mistake is in the atan(x2/x1) which in Deb's paper is written as
        atan(x/y) by accident.
        The problem is good, as the objective space is the same as the decision variable space. In Deb's paper some properties
        about it are explained, such as where the Front lies and the non-convexity of the Pareto front.
        :return:
        """
        return [x[0], x[1]]

    @staticmethod
    def OSY(x):
        """
        From paper:
        "K. Deb, A. Pratap, T. Meyarivan - Constrained Test Problems for Multi-objective Evolutionary Optimization"
        On page 4, the OSY problem is introduced, which is orignally presented in paper:
        "Osyczka A., Kundu S. (1995) -  A new method to solve generalized multicriteria optimization problems using the simple genetic algorithm."
        The problem is very interesting, since it is nonlinear and has 6 decision variables, 2 objectives and a non
        convex Pareto front.
        Deb explains within detail how to obtain the Pareto front of this problem.
        :return:
        """
        f1 = -25 * math.pow(x[0] - 2, 2) - math.pow(x[1] - 2, 2) - math.pow(x[2] - 1, 2) - math.pow(x[3] - 4, 2) - math.pow(x[4] - 1, 2)
        f2 = math.pow(x[0], 2) + math.pow(x[1], 2) + math.pow(x[2], 2) + math.pow(x[3], 2) + math.pow(x[4], 2) + math.pow(x[5], 2)
        return [f1, f2]


class MOO_Constraints:
    @staticmethod
    def BK1_constraint(x):
        """
        x1 € [-5, 10]
        x2 € [-5, 10]
        :param x: np.ndarray
        :return: <int> number of constraints violated
        """
        return np.count_nonzero((x < -5) | (x > 10))


