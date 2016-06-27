import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class State:
    dealercard = random.randint(1,10)
    playersum = random.randint(1,10)

# adds some drawn card value
def drawcard(current):
    if random.randint(1,3) < 3:
        current += random.randint(1,10)
    else:
        current -= random.randint(1,10)
    return current

# action {0: stick, 1: hit}
def step(state, action):
    if action == 1:
        state.playersum = drawcard(state.playersum)
        if state.playersum < 1 or state.playersum > 21:
            return "terminal", -1
        else:
            return state, 0
    elif action == 0:
        while(state.dealercard < 17):
            state.dealercard = drawcard(state.dealercard)
            if state.dealercard < 1 or state.dealercard > 21:
                return "terminal", 1
        if state.dealercard > state.playersum:
            return "terminal", -1
        elif state.dealercard < state.playersum:
            return "terminal", 1
        else:
            return "terminal", 0


# value[action][dealercard][playersum] is the value function, table lookup should work with so little states


def montecarlo(value,counter):
    state = State()
    state.dealercard = random.randint(1,10)
    state.playersum = random.randint(1,10)
    totalreward = 0
    visits = []
    while state != "terminal":
        action = None
        e = 100.0 / (100.0 + np.sum(counter[:,state.dealercard, state.playersum],axis=0))
        if (random.random() < e):
            action = random.randint(0,1)
        else:
            action = np.argmax(value[:, state.dealercard, state.playersum])
        counter[action, state.dealercard, state.playersum] += 1
        visits.append((action, state.dealercard, state.playersum))
        state, reward = step(state, action)
        totalreward += reward
    # since the only reward is at the end, doesn't matter when we visited the state
    for action, dealercard, playersum in visits:
        a = 1 / counter[action,dealercard,playersum]
        g = totalreward
        # print "section"
        # print a*(g - value[action,dealercard,playersum])
        value[action,dealercard,playersum] = value[action,dealercard,playersum] + a*(g - value[action,dealercard,playersum])
        # print value[action,dealercard,playersum]
    return value, counter

value = np.zeros((2,11,22))
counter = np.zeros((2,11,22))
for i in xrange(100000):
    vale, counter = montecarlo(value,counter)

bestval = np.amax(value,axis=0)
fig = plt.figure()
ha = fig.add_subplot(111, projection='3d')
x = range(11)
y = range(22)
X, Y = np.meshgrid(y, x)
# print np.shape(bestval)
# print bestval
ha.plot_wireframe(X, Y, bestval)
plt.show()
