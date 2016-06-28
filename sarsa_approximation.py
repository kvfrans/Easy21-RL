import random
import numpy as np

from environment import *


def app_dealercard(dealercard):
    containers = []
    if dealercard in [1,2,3,4]:
        containers.append(0)
    if dealercard in [4,5,6,7]:
        containers.append(1)
    if dealercard in [7,8,9,10]:
        containers.append(2)
    return containers

def app_playersum(playersum):
    containers = []
    if playersum in [1,2,3,4,5,6]:
        containers.append(0)
    if playersum in [4,5,6,7,8,9]:
        containers.append(1)
    if playersum in [7,8,9,10,11,12]:
        containers.append(2)
    if playersum in [10,11,12,13,14,15]:
        containers.append(3)
    if playersum in [13,14,15,16,7,18]:
        containers.append(4)
    if playersum in [16,17,18,19,20,21]:
        containers.append(5)
    return containers

def getvalue(value,action,dealercard,playersum):
    total = 0
    for s1 in app_dealercard(dealercard):
        for s2 in app_playersum(playersum):
            total += value[action,s1,s2]
    return total;


# scaling = lambda
def sarsa(value,scaling):
    # print "sarsa"
    state = State()
    state.dealercard = random.randint(1,10)
    state.playersum = random.randint(1,10)

    eligibility = np.zeros((2,3,6))

    while state != "terminal":
        # print "state is"
        # print state
        action = None
        e = 0.05
        if (random.random() < e):
            action = random.randint(0,1)
        else:
            action = 0 if getvalue(value,0,state.dealercard,state.playersum) > getvalue(value,1,state.dealercard,state.playersum) else 1

        for s1 in app_dealercard(state.dealercard):
            for s2 in app_playersum(state.playersum):
                eligibility[action, s1, s2] += 1
        old_dealercard = state.dealercard
        old_playersum = state.playersum
        state, reward = step(state, action)
        # a hack to get around dividing by zero errors
        a = 0.01

        g_scalar = reward
        if state == "terminal":
            g_scalar += 0
        else:
            g_scalar += max(getvalue(value,0,state.dealercard,state.playersum), getvalue(value,1,state.dealercard,state.playersum))
        # print g_scalar
        g = g_scalar - value
        # print a*g*eligibility
        value = value + a*g*eligibility
        # a = 1.0 / counter[action,old_dealercard,old_playersum]
        # value[action,old_dealercard,old_playersum] = value[action,old_dealercard,old_playersum] + a*(g_scalar - value[action,old_dealercard,old_playersum])

        # decrease eligiblity traces by lambda scaling
        eligibility = eligibility * scaling


    return value
