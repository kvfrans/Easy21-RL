import random
import numpy as np

from environment import *

# scaling = lambda
def sarsa(value,counter,scaling):
    # print "sarsa"
    state = State()
    state.dealercard = random.randint(1,10)
    state.playersum = random.randint(1,10)

    eligibility = np.zeros((2,11,22))

    while state != "terminal":
        # print "state is"
        # print state
        action = None
        e = 100.0 / (100.0 + np.sum(counter[:,state.dealercard, state.playersum],axis=0))
        if (random.random() < e):
            action = random.randint(0,1)
        else:
            action = np.argmax(value[:, state.dealercard, state.playersum])
        eligibility[action, state.dealercard, state.playersum] += 1
        old_dealercard = state.dealercard
        old_playersum = state.playersum
        counter[action, state.dealercard, state.playersum] += 1
        state, reward = step(state, action)
        # a hack to get around dividing by zero errors
        counter[counter == 0] = -1
        a = 1.0 / counter
        a[counter == -1] = 0
        counter[counter == -1] = 0

        g_scalar = reward
        if state == "terminal":
            g_scalar += 0
        else:
            g_scalar += np.amax(value[:,state.dealercard,state.playersum],axis=0)
        # print g_scalar
        g = g_scalar - value
        # print a*g*eligibility
        value = value + a*g*eligibility
        # a = 1.0 / counter[action,old_dealercard,old_playersum]
        # value[action,old_dealercard,old_playersum] = value[action,old_dealercard,old_playersum] + a*(g_scalar - value[action,old_dealercard,old_playersum])

        # decrease eligiblity traces by lambda scaling
        eligibility = eligibility * scaling


    return value, counter
