import random
import numpy as np

from environment import *

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
