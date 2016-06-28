import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sarsa_eligibility import *
from montecarlo import *

# value[action][dealercard][playersum] is the value function, table lookup should work with so little states
value = np.zeros((2,11,22))
counter = np.zeros((2,11,22))

# run monte-carlo
# for i in xrange(1000000):
#     value, counter = montecarlo(value,counter)

# np.save("montecarlo.npy",value)
value = np.load("montecarlo.npy")

testlooking = None

# calculate MSQ per lambda
msq = np.zeros(11)
for k in xrange(11):
    td_value = np.zeros((2,11,22))
    td_counter = np.zeros((2,11,22))
    for i in xrange(1000):
        td_value, td_counter = sarsa(td_value,td_counter,k*0.1)
    msq[k] = np.sum(np.square(value - td_value)) / (2.0*11*22)
    testlooking = td_value

#calculate MSQ for lambda 0 & 1, over episodes
msq = np.zeros((2,100))
for k in xrange(2):
    td_value = np.zeros((2,11,22))
    td_counter = np.zeros((2,11,22))
    for i in xrange(100000):
        td_value, td_counter = sarsa(td_value,td_counter,k*1)
        if i % 1000 == 0:
            msq[k,(i / 1000)] = np.sum(np.square(value - td_value)) / (2.0*11*22)



# plot monte-carlo value func
# bestval = np.amax(value,axis=0)
# bestval = np.amax(testlooking,axis=0)
# fig = plt.figure()
# ha = fig.add_subplot(111, projection='3d')
# x = range(10)
# y = range(21)
# X, Y = np.meshgrid(y, x)
# ha.plot_wireframe(X+1, Y+1, bestval[1:,1:])
# ha.set_ylabel("dealer starting card")
# ha.set_xlabel("player current sum")
# ha.set_zlabel("value of state")

# # plot msq on lambda scaling
# plt.plot(np.arange(11) * 0.1,msq)
# plt.ylabel("Mean Squared Error")
# plt.xlabel("lambda scaling")

# plot msq on episodes
plt.plot(np.arange(100) * 1000,msq[0],label="lambda 0")
plt.plot(np.arange(100) * 1000,msq[1],label="lambda 1")
plt.ylabel("Mean Squared Error")
plt.xlabel("episodes")

plt.show()
