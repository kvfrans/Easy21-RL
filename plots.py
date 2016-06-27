import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from montecarlo import *

# value[action][dealercard][playersum] is the value function, table lookup should work with so little states
value = np.zeros((2,11,22))
counter = np.zeros((2,11,22))

# run monte-carlo
for i in xrange(100000):
    vale, counter = montecarlo(value,counter)


# plot
bestval = np.amax(value,axis=0)
fig = plt.figure()
ha = fig.add_subplot(111, projection='3d')
x = range(11)
y = range(22)
X, Y = np.meshgrid(y, x)
ha.plot_wireframe(X, Y, bestval)
ha.set_xlabel("dealer starting card")
ha.set_ylabel("player current sum")
ha.set_zlabel("value of state")

plt.show()
