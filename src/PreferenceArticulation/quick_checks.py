import numpy as np
import math as math
import plotly.graph_objects as go


f1 = [0.9457533241109305, 0.8399240327613827, 0.6321205588285577, 0.34156746712171393, 0.08220978425157577]
aspirations = [[f1[i], f1[-i-1]] for i in range(5)]
print(aspirations)