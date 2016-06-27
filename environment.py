import random
import numpy as np

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
