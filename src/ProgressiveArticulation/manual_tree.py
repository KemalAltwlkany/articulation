import numpy as np
import os as os
import pandas as pd



# RBDT - no implementation was found on the internet.
# Might implement it myself (the RBDT)
# Entire class is hardcoded due to lack of time for implementing the RBDT algorithm.
class RuleBasedTree:
    def __init__(self, d=None):
        # d is a list of dictionaries.
        self.f1_A = d[0]['A']
        self.f1_B = d[0]['B']
        self.f1_C = d[0]['C']
        self.f1_D = d[0]['D']
        self.f1_E = d[0]['E']
        self.f1_F = d[0]['F']

        self.f2_A = d[0]['A']
        self.f2_B = d[0]['B']
        self.f2_C = d[0]['C']
        self.f2_D = d[0]['D']
        self.f2_E = d[0]['E']
        self.f2_F = d[0]['F']

    def classify(self, y):
        # Returns class of solution based on the values of its criteria
        f1, f2 = y[0], y[1]
        if f1 > self.f1_C and f2 > self.f2_C:
            return 'I12'
        elif f1 < self.f1_C and f2 < self.f2_C:
            return 'G'
        elif f1 > self.f1_D:
            return 'I1*'
        elif f2 > self.f2_D:
            return 'I2*'
        elif f1 > self.f2_C:
            return 'I1'
        else:
            return 'I2'


def save




