import sys as sys
import json as json
import os as os
import random as random

"""
This script simply creates a python dictionary and dumps it to a specified location as a json file.
I'm using this, so that I can easily generate a few json formatted settings, and then load them when doing tests.
"""

# Specify save folder location for json file, and file name
save_folder = "/home/kemal/Programming/Python/Articulation/Tests/standardized_tests/TS/FON"
file_name = "standard_0"

# Description for test
description = "Standard test 0."

# Specify values here
delta = 0.02
max_iter = 1000
tabu_list_max_length = 20
M = 100
max_loops = 30
min_progress = delta/100.
weights = [0.5, 0.5]
random.seed(0)
init_sol = [random.uniform(-4, 4), random.uniform(-4, 4), random.uniform(-4, 4)]



# ---- script does the rest ---------
try:
    os.chdir(save_folder)
except OSError:
    print('Could not change to provided directory. Exiting with flag 17.')
    sys.exit(17)

data = {}
data['delta'] = delta
data['max_iter'] = max_iter
data['tabu_list_max_length'] = tabu_list_max_length
data['M'] = M
data['max_loops'] = max_loops
data['min_progress'] = min_progress
data['weights'] = weights
data['init_sol'] = init_sol
data['description'] = description

with open(file_name + ".txt", 'w') as output:
    json.dump(data, output)
