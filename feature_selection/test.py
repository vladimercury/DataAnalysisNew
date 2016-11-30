import numpy as np
import scipy.stats as stats
import math
r = -0.17575757575
v = 8
t = r * math.sqrt(v) / math.sqrt(1 - r ** 2)
print(stats.t.sf(np.abs(t), v) * 2)