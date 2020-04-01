from functools import reduce

import numpy as np
import matplotlib.pyplot as plt
import math as math
# from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
# import plotly.express as px
# from decimal import Decimal
import random as random
# from scipy.spatial import Delaunay
# import matplotlib.cm as cm
# import chart_studio.plotly as py


fig_num = 1


# S. Huband, 7 recommendations:
# R1.) No Extremal Parameters
# R2.) No Medial Parameters
# R3.) Scalable Number of Parameters
# R4.) Scalable Number of Objectives
# R5.) Dissimilar Parameter Domains
# R6.) Dissimilar Tradeoff Ranges
# R7.) Pareto Optima Known

# done
# noinspection DuplicatedCode
def BK1():
    """
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

    :return:
    """
    # first part plots the entire scatter plot, the grid of the fitness space
    global fig_num
    x1 = np.linspace(-5, 10, 300)
    x2 = np.linspace(-5, 10, 300)
    f1 = []
    f2 = []
    for i in x1:
        for j in x2:
            f1.append(i ** 2 + j ** 2)
            f2.append((i - 5) ** 2 + (j - 5) ** 2)

    plt.figure(fig_num)
    plt.scatter(f1, f2)
    plt.xlabel('f1(x1, x2)')
    plt.ylabel('f2(x1, x2)')
    plt.title('Evaluation of f1(x1, x2), f2(x1, x2)')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True)
    fig_num = fig_num + 1

    # second part plots the Pareto front of the problem
    x1 = np.linspace(0, 5, 50)
    x2 = np.linspace(0, 5, 50)
    f1 = x1**2 + x2**2
    f2 = (x1-5)**2 + (x2-5)**2
    plt.plot(f1, f2, linewidth=3.5, linestyle='-', color='r')
    plt.figure(fig_num)
    plt.plot(f1, f2, linewidth=3.5)
    plt.xlabel('f1(x1, x2)')
    plt.ylabel('f2(x1, x2)')
    plt.title('Pareto front of f1(x1, x2), f2(x1, x2)')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True)
    fig_num = fig_num + 1
    plt.show()


# done
# noinspection DuplicatedCode
def IM1():
    """
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

    :return:
    """
    # first part plots the entire scatter plot, the grid of the fitness space
    global fig_num
    x1 = np.linspace(1, 4, 300)
    x2 = np.linspace(1, 2, 300)
    f1 = []
    f2 = []
    for i in x1:
        for j in x2:
            f1.append(2*math.sqrt(i))
            f2.append(i*(1-j) + 5)

    plt.figure(fig_num)
    plt.scatter(f1, f2)
    plt.xlabel('f1(x1, x2)')
    plt.ylabel('f2(x1, x2)')
    plt.title('Evaluation of f1(x1, x2), f2(x1, x2)')
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.grid(True)
    fig_num = fig_num + 1

    # second part plots the Pareto front of the problem

    f1 = 2*np.sqrt(x1)
    x2 = 2
    f2 = x1*(1-x2) + 5
    plt.plot(f1, f2, linewidth=3.5, linestyle='-', color='r')
    plt.figure(fig_num)
    plt.plot(f1, f2, linewidth=3.5, linestyle='-', color='b')
    plt.xlabel('f1(x1, x2)')
    plt.ylabel('f2(x1, x2)')
    plt.title('Pareto front of f1(x1, x2), f2(x1, x2)')
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.grid(True)
    fig_num = fig_num + 1
    plt.show()


# not used, not enough information.
# noinspection DuplicatedCode
def a3():
    """
    In the work of S. Huband, this function is presented. It originates from:
    "https://link.springer.com/chapter/10.1007/BFb0029752"
    "F.. Kursawe, “A variant of evolution strategies for vector optimization,”
        in Lecture Notes in Computer Science, H.-P. Schwefel and R. Männer,
        Eds. Berlin, Germany: Springer-Verlag, 1991, vol. 496, Proc. Parallel
        Problem Solving From Nature. 1st Workshop, PPSN I, pp. 193–197."
    I possess this paper, however it only presents this function without devoting any time to analyze the details
    about it. The Pareto optimal set is not given either.
    :return:
    """
    global fig_num
    x1_space = np.linspace(-5, 5, 200)
    x2_space = np.linspace(-5, 5, 200)
    x3_space = np.linspace(-5, 5, 200)
    f1 = []
    f2 = []
    for x1 in x1_space:
        for x2 in x2_space:
            for x3 in x3_space:
                s1 = -10*math.exp(-0.2*math.sqrt(x1**2 + x2**2)) - 10*math.exp(-0.2*math.sqrt(x2**2+x3**2))
                f1.append(s1)
                s2 = math.pow(math.fabs(x1), 0.8) + math.pow(math.fabs(x2), 0.8) + math.pow(math.fabs(x3), 0.8)
                s3 = math.sin(math.pow(x1, 3)) + math.sin(math.pow(x2, 3)) + math.sin(math.pow(x3, 3))
                f2.append(s2+5*s3)

    plt.figure(fig_num)
    plt.scatter(f1, f2)
    plt.xlabel('f1(x1, x2, x3)')
    plt.ylabel('f2(x1, x2, x3)')
    plt.title('Evaluation of f1(x1, x2, x3), f2(x1, x2, x3)')
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.grid(True)
    fig_num = fig_num + 1
    plt.show()


# done
# noinspection DuplicatedCode
# noinspection PyPep8Naming
def SCH1():
    """
    SCH1 - Schaffers test function, from Schaffer 1984, I cited it from:
    "K. Deb, L. Thiele, M. Laumanns, E. Zitzler - Scalable test problems for evolutionary multi-objective optimization"
    The Pareto optimal front can be obtained for any:
    x € [0, 2] (Pareto optimal set).
    :return:
    """
    global fig_num
    x1_space = np.linspace(-10, 10, 300)
    f1 = []
    f2 = []
    for x1 in x1_space:
        f1.append(x1**2)
        f2.append((x1-2)**2)

    plt.figure(fig_num)
    plt.scatter(f1, f2)
    plt.xlabel('f1(x1)')
    plt.ylabel('f2(x1)')
    plt.title('Evaluation of f1(x1), f2(x1)')
    plt.xlim(0, 6)
    plt.ylim(0, 6)
    plt.grid(True)
    fig_num = fig_num + 1

    # second part plots the Pareto front of the problem
    x1 = np.linspace(0, 2, 100)
    f1 = x1**2
    f2 = (x1-2)**2
    plt.plot(f1, f2, linewidth=3.5, linestyle='-', color='r')
    plt.figure(fig_num)
    plt.plot(f1, f2, linewidth=3.5)
    plt.xlabel('f1(x1)')
    plt.ylabel('f2(x1)')
    plt.title('Pareto front of f1(x1), f2(x1)')
    plt.xlim(0, 6)
    plt.ylim(0, 6)
    plt.grid(True)
    fig_num = fig_num + 1
    plt.show()


# done
# noinspection DuplicatedCode
def FON():
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
    # DEMONSTRATES A 3D SEARCH SPACE.
    global fig_num
    x_space = np.linspace(-4, 4, 100)  # same for all variables xi, for i=1,...,n
    f1 = []
    f2 = []
    p = 1./math.sqrt(3.)
    for x1 in x_space:
        for x2 in x_space:
            for x3 in x_space:
                sum1 = math.pow(x1 - p, 2) + math.pow(x2 - p, 2) + math.pow(x3 - p, 2)
                sum2 = math.pow(x1 + p, 2) + math.pow(x2 + p, 2) + math.pow(x3 + p, 2)
                f1.append(1 - math.exp(-sum1))
                f2.append(1 - math.exp(-sum2))

    plt.figure(fig_num)
    plt.scatter(f1, f2)
    plt.xlabel('f1(x1, x2, x3)')
    plt.ylabel('f2(x1, x2, x3)')
    plt.title('Evaluation of f1(x1, x2, x3), f2(x1, x2, x3)')
    plt.xlim(0, 1.2)
    plt.ylim(0, 1.2)
    plt.grid(True)
    fig_num = fig_num + 1

    # second part plots the Pareto front of the problem
    x_space = np.linspace(-p, p, 100)
    f1 = []
    f2 = []
    for x1 in x_space:
        x2 = x1
        x3 = x1
        sum1 = math.pow(x1 - p, 2) + math.pow(x2 - p, 2) + math.pow(x3 - p, 2)
        sum2 = math.pow(x1 + p, 2) + math.pow(x2 + p, 2) + math.pow(x3 + p, 2)
        f1.append(1 - math.exp(-sum1))
        f2.append(1 - math.exp(-sum2))

    plt.plot(f1, f2, linewidth=3.5, linestyle='-', color='r')
    plt.figure(fig_num)
    plt.plot(f1, f2, linewidth=3.5)
    plt.xlabel('f1(x1, x2, x3)')
    plt.ylabel('f2(x1, x2, x3)')
    plt.title('Pareto front of f1(x1, x2, x3), f2(x1, x2, x3)')
    plt.xlim(0, 1.2)
    plt.ylim(0, 1.2)
    plt.grid(True)
    fig_num = fig_num + 1
    plt.show()


# done
# noinspection DuplicatedCode
def comet_problem():
    """
    Paper: "K. Deb, L. Thiele, M. Laumanns, E. Zitzler - Scalable test problems for evolutionary multi-objective optimization"
    introduces the Comet problem in section 4.6, on page 10.
    It is an illustration of the bottom-up approach to design test problems.
    It is a tri-objective problem, and the Pareto front is known and explained within detail in the paper itself.
    :return:
    """
    global fig_num
    x1_space = np.linspace(1, 3.5, 100)
    x2_space = np.linspace(-2, 2)
    f1 = []
    f2 = []
    f3 = []
    f1red, f2red, f3red = [], [], []
    pts = 0
    while pts < 3000:
        x1 = random.uniform(1, 3.5)
        x2 = random.uniform(-2, 2)
        val1 = math.pow(x1, 3)*x2
        if val1 - 2 > 0 or val1 + 2 < 0:
            p2 = math.pow(x1, 3) * math.pow(x2, 2)
            f1.append(-(p2 - 10 * x1 - 4 * x2))
            f2.append(-(p2 - 10 * x1 + 4 * x2))
            f3.append(-3 * math.pow(x1, 2))
            continue
        else:
            pts = pts + 1
            p2 = math.pow(x1, 3) * math.pow(x2, 2)
            f1red.append(-(p2 - 10*x1 - 4*x2))
            f2red.append(-(p2 - 10*x1 + 4*x2))
            f3red.append(-3*math.pow(x1, 2))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(f1, f2, f3)
    ax.scatter(f1red, f2red, f3red, color='r')

    fig2 = go.Figure(data=[go.Scatter3d(
        x=f1,
        y=f2,
        z=f3,
        mode='markers',
        marker=dict(
            size=6,
            color=f3,  # set color to an array/list of desired values
            colorscale='Viridis',  # choose a colorscale
            opacity=0.4
        )
    )])
    fig2 = go.Figure(data=[go.Scatter3d(
        x=f1red,
        y=f2red,
        z=f3red,
        mode='markers',
        marker=dict(
            size=6,
            color=f3red,  # set color to an array/list of desired values
            colorscale='magma',  # choose a colorscale
            opacity=0.8
        )
    )])
    fig2.show()
    plt.show()

    # x1_space = np.linspace(-1, 3.5, 25)
    # x2_space = np.linspace(-2, 2, 25)
    # x3_space = np.linspace(0, 1, 25)
    # f1 = []
    # f2 = []
    # f3 = []
    # for x1 in x1_space:
    #     for x2 in x2_space:
    #         for x3 in x3_space:
    #             p1 = 1 + x3
    #             p2 = math.pow(x1, 3) * math.pow(x2, 2)
    #             f1.append(-p1 * (p2 - 10*x1 - 4*x2))
    #             f2.append(-p1 * (p2 - 10*x1 + 4*x2))
    #             f3.append(-3*p1*math.pow(x1, 2))



    # fig = go.Figure(data=[go.Surface(z=np.array(f3), y=np.array(f2), x=np.array(f1))])



    # fig = go.Figure(data=[go.Mesh3d(x=f1,
    #                                 y=f2,
    #                                 z=f3,
    #                                 opacity=0.5,
    #                                 color='rgba(244,22,100,0.6)'
    #                                 )])
    #
    # fig.update_layout(
    #     scene=dict(
    #         xaxis=dict(nticks=4, range=[-10, 40], ),
    #         yaxis=dict(nticks=4, range=[-10, 40], ),
    #         zaxis=dict(nticks=5, range=[-40, 0], ), ))


    # fig.show()
    # fig = plt.figure(fig_num)
    # ax = plt.axes(projection='3d')
    # X, Y = np.meshgrid(x1_space, x2_space)
    # Z = np.array(f3)
    # ax.contour3D(f1, f2, f3, 50, cmap='binary')
    # ax.set_xlabel('-f1')
    # plt.xlim(-10, 40)
    # plt.ylim(-10, 40)
    # plt.zlim(-40, 0)
    # ax.set_ylabel('-f2')
    # ax.set_zlabel('-f3')

    # ax.plot_trisurf(f1, f2, f3, cmap='viridis', edgecolor='none')
    # plt.xlabel('f1(x1)')
    # plt.ylabel('f2(x1)')
    # plt.title('Evaluation of f1(x1), f2(x1)')
    # plt.xlim(0, 6)
    # plt.ylim(0, 6)
    # plt.grid(True)
    # fig_num = fig_num + 1
    # plt.show()


# done
def TNK():
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
    global fig_num
    x1_space = np.linspace(0, math.pi, 5000)  # should be extra dense, because of the non-convex border
    x2_space = np.linspace(0, math.pi, 5000)
    f1 = []
    f2 = []
    for x1 in x1_space:
        for x2 in x2_space:
            if math.isclose(x1, 0):
                c1 = math.pow(x2, 2) - 1.1
            else:
                c1 = math.pow(x1, 2) + math.pow(x2, 2) - 1 - 0.1*math.cos(16*math.atan(x2/x1))
            c2 = math.pow(x1 - 0.5, 2) + math.pow(x2 - 0.5, 2) - 0.5
            if c1 >= 0 >= c2:
                f1.append(x1)
                f2.append(x2)
    fig = plt.figure(fig_num)
    fig_num = fig_num + 1
    plt.scatter(f1, f2)
    plt.xlabel('f1(x1) = x1')
    plt.ylabel('f2(x2) = x2')
    plt.title('Evaluation of f1(x1), f2(x2)')
    plt.xlim(0, 1.2)
    plt.ylim(0, 1.2)
    plt.grid(True)
    fig_num = fig_num + 1

    # second part plots the Pareto front of the problem
    f1 = []
    f2 = []
    for x1 in x1_space:
        for x2 in x2_space:
            if math.isclose(x1, 0):
                c1 = math.pow(x2, 2) - 1.1
            else:
                c1 = math.pow(x1, 2) + math.pow(x2, 2) - 1 - 0.1 * math.cos(16 * math.atan(x2 / x1))
            c2 = math.pow(x1 - 0.5, 2) + math.pow(x2 - 0.5, 2) - 0.5
            if math.isclose(c1, 0, abs_tol=0.001) and 0.0 >= c2:
                f1.append(x1)
                f2.append(x2)

    f1_pom = []
    f2_pom = []
    # this additional nested for-loop is a brute-force method that filters out the actual Pareto optimums.
    # the reason for this is that even though the previous piece of code does find the curve which defines the
    # region border, its non convex and not the entire curve makes up the Pareto front.
    for i in range(len(f1)):
        p1, p2 = f1[i], f2[i]
        trig = False
        for j in range(len(f1)):
            if p1 - f1[j] > 0 and p2 - f2[j] > 0:
                trig = True
                break
            else:
                continue
        if not trig:
            f1_pom.append(p1)
            f2_pom.append(p2)

    f1 = f1_pom
    f2 = f2_pom

    plt.scatter(f1, f2, linewidth=3.5, linestyle='-', color='r')
    plt.figure(fig_num)
    plt.scatter(f1, f2, linewidth=3.5)
    plt.xlabel('f1(x1, x2)')
    plt.ylabel('f2(x1, x2)')
    plt.title('Pareto front of f1(x1, x2), f2(x1, x2)')
    plt.xlim(0, 1.2)
    plt.ylim(0, 1.2)
    plt.grid(True)
    fig_num = fig_num + 1
    plt.show()


# done
def OSY():
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
    global fig_num
    f1 = []
    f2 = []
    random.seed(0)
    while len(f1) < 5*1e5:
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 10)
        x3 = random.uniform(1, 5)
        x4 = random.uniform(0, 6)
        x5 = random.uniform(0, 5)
        x6 = random.uniform(0, 10)
        if x1 + x2 - 2 < 0:
            continue
        if 6 - x1 - x2 < 0:
            continue
        if 2 + x1 - x2 < 0:
            continue
        if 2 - x1 + 3*x2 < 0:
            continue
        if 4 - math.pow(x3 - 3, 2) - x4 < 0:
            continue
        if math.pow(x5 - 3, 2) + x6 - 4 < 0:
            continue
        f1.append(-25*math.pow(x1 - 2, 2) - math.pow(x2 - 2, 2) - math.pow(x3 - 1, 2) - math.pow(x4 - 4, 2) - math.pow(x5 - 1, 2))
        f2.append(x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2)
        # print(len(f1))

    fig = plt.figure(fig_num)
    fig_num = fig_num + 1
    plt.scatter(f1, f2)
    plt.xlabel('f1(x1, x2, x3, x4, x5)')
    plt.ylabel('f2(x1, x2, x3, x4, x5, x6')
    plt.title('Evaluation of f1, f2')
    plt.xlim(-300, 0)
    plt.ylim(0, 80)
    plt.grid(True)
    fig_num = fig_num + 1

    f1 = []
    f2 = []
    x4, x6 = 0, 0
    # region AB
    x1, x2, x5 = 5, 1, 5
    x3_space = np.linspace(1, 5, 100)
    for x3 in x3_space:
        f1.append(-25*math.pow(x1 - 2, 2) - math.pow(x2 - 2, 2) - math.pow(x3 - 1, 2) - 16 - math.pow(x5 - 1, 2))
        f2.append(x1**2 + x2**2 + x3**2 + x5**2)

    # region BC
    x1, x2, x5 = 5, 1, 1
    x3_space = np.linspace(1, 5, 100)
    for x3 in x3_space:
        f1.append(-25*math.pow(x1 - 2, 2) - math.pow(x2 - 2, 2) - math.pow(x3 - 1, 2) - 16 - math.pow(x5 - 1, 2))
        f2.append(x1**2 + x2**2 + x3**2 + x5**2)

    # region CD
    x1_space = np.linspace(4.056, 5, 50)
    x3, x5 = 1, 1
    for x1 in x1_space:
        x2 = (x1 - 2.)/3.
        f1.append(-25*math.pow(x1 - 2, 2) - math.pow(x2 - 2, 2) - math.pow(x3 - 1, 2) - 16 - math.pow(x5 - 1, 2))
        f2.append(x1**2 + x2**2 + x3**2 + x5**2)

    # region DE
    x1, x2, x5 = 0, 2, 1
    x3_space = np.linspace(1, 3.732, 100)
    for x3 in x3_space:
        f1.append(-25*math.pow(x1 - 2, 2) - math.pow(x2 - 2, 2) - math.pow(x3 - 1, 2) - 16 - math.pow(x5 - 1, 2))
        f2.append(x1**2 + x2**2 + x3**2 + x5**2)

    # region EF
    x1_space = np.linspace(0, 1, 50)
    x3, x5 = 1, 1
    for x1 in x1_space:
        x2 = 2 - x1
        f1.append(-25*math.pow(x1 - 2, 2) - math.pow(x2 - 2, 2) - math.pow(x3 - 1, 2) - 16 - math.pow(x5 - 1, 2))
        f2.append(x1**2 + x2**2 + x3**2 + x5**2)

    plt.scatter(f1, f2, linewidth=3.5, linestyle='-', color='r')
    plt.figure(fig_num)
    plt.scatter(f1, f2, linewidth=3.5)
    plt.xlabel('f1(x1, x2)')
    plt.ylabel('f2(x1, x2)')
    plt.title('Pareto front of f1(x1, x2), f2(x1, x2)')
    plt.xlim(-300, 0)
    plt.ylim(0, 80)
    plt.grid(True)
    fig_num = fig_num + 1
    plt.show()


if __name__ == '__main__':
    BK1()
    # IM1()
    # SCH1()
    # FON()
    # TNK()
    # OSY()
    # comet_problem()
    # a3()
