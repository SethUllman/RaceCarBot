import numpy as np
import pandas as pd
import random

class ValueIteration:
    def __init__(self, car, track, memory):
        self.car = car
        self.track = track
        if memory is not None and not memory.empty:
            self.values = memory     #Q-Table
        else:
            indexTuples = [(row, col, x, y) for row in range(0, self.track.row) for col in range(0, self.track.col) for x in range(-5, 6) for y in range(-5, 6)]
            labels = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
            self.values = pd.DataFrame(index=indexTuples, columns=labels, dtype=float)
            self.values.fillna(0, inplace=True)

    def valueIteration(self, bellmanError, discount):
        delta = float('inf')

        while delta > bellmanError:
            delta = 0
            max_value = float('-inf')

            states = self.values.index.tolist()
            actions = self.values.columns.tolist()
            for state in states:

                for action in actions:
                    v = self.values.at[state, action]
                    value = reward(state, action) + discount*getNextState(state, action) # needs to be a value
                    
                    if value > max_value:
                        max_value = value


                # V = 

        # δ = float('inf')  # Initialize delta to positive infinity

        # while δ > ε:
        #     δ = 0  # Reset delta for the current iteration

        #     for each state s in S:
        #         v = V[s]
        #         max_value = -∞

        #         for each action a in A:
        #             next_state = get_next_state(s, a)
        #             value = R(s, a) + γ * V[next_state]
        #             max_value = max(max_value, value)

        #         V[s] = max_value
        #         δ = max(δ, abs(v - V[s]))

        def reward(self, state, action):
            return
        
        def getNextState(self, state, action):
            return