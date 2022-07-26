from scipy import stats
import numpy as np

a = np.array([0,0,0,0,0,0,1])
b = a + 10
c = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])

print(a, b, min(a))

print(stats.entropy(a), stats.entropy(b), stats.entropy(c))