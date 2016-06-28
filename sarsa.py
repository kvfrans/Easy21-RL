import random
import numpy as np
import math

from environment import *

# scaling = lambda
def sarsa(value,counter,scaling):
    state = State()
    state.dealercard = random.randint(1,10)
    state.playersum = random.randint(1,10)
    transitions = []
    while state != "terminal":
        action = None
        e = 100.0 / (100.0 + np.sum(counter[:,state.dealercard, state.playersum],axis=0))
        if (random.random() < e):
            action = random.randint(0,1)
        else:
            action = np.argmax(value[:, state.dealercard, state.playersum])
        counter[action, state.dealercard, state.playersum] += 1
        old_dealercard = state.dealercard
        old_playersum = state.playersum
        state, reward = step(state, action)
        # i took action A at state S, and got a reward of R.
        # print "i took action %d at state (%d,%d) and got reward %d" % (action, old_dealercard, old_playersum, reward)
        if state != "terminal":
            transitions.append((action, old_dealercard, old_playersum, reward, state.dealercard, state.playersum))
        else:
            transitions.append((action, old_dealercard, old_playersum, reward, -1, -1))

    # for every state that we visited
    for index, item in enumerate(transitions):
        action, dealercard, playersum, reward, _, _ = item
        rewardsum = 0
        g_total = 0
        stepstillend = len(transitions) - index
        # print "updating on state %d, %d" % (dealercard, playersum)
        # for all the states that we saw afterwards
        for lookahead in xrange(stepstillend):
            # print "looking ahead %d steps" % (lookahead+1)
            nextstatereward = None
            # the state we're observing
            lookaheadstate = transitions[lookahead + index]

            if lookaheadstate[4] == -1:
                # no value for being in terminal state
                nextstatevalue = 0
            else:
                # value of taking optimal action from next state
                nextstatevalue = np.argmax(value[:, lookaheadstate[4], lookaheadstate[5]])

            # add reward that got from this transition
            rewardsum = rewardsum + lookaheadstate[3]

            # bootstrapped new value: rewards until the state + the value of the state
            g = rewardsum + nextstatevalue
            # print "new value should be %f" % g
            valueadded = None
            if lookahead == stepstillend - 1:
                # different weight for terminal state, that simulates being in that state forever
                valueadded = math.pow(scaling, stepstillend - 1)*g
                # print "mathpow %f" % math.pow(scaling, stepstillend - 1)
                # print "scaling %f" % scaling
                # print "stepstillend -1 %d" % (stepstillend -1)
            else:
                # geometric weighting for each step forward
                valueadded = (1.0 - scaling)*math.pow(scaling, lookahead)*g
            # print "increasing total by %f" % valueadded
            g_total += valueadded

        # step size
        a = 1.0 / counter[action,dealercard,playersum]
        # print "in total, value should be %f" % g_total
        # update the value function towards the new iteration
        value[action,dealercard,playersum] = value[action,dealercard,playersum] + a*(g_total - value[action,dealercard,playersum])

    return value, counter
