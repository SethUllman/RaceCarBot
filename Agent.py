import numpy as np
import pandas as pd
from Car import Car
from Track import Track
from ValueIteration import ValueIteration
from QLearning import QLearning
from SARSA import SARSA

# ---------------- Agent ----------------
class Agent:

    def __init__(self, filename, memory = None):
        self.car = Car()
        self.track = Track()
        self.track.parseTrack(filename)
        if memory is None:
            self.memory = pd.DataFrame()
        else:
            self.memory = pd.read_csv(memory)

    def valueIteration(self, bellmanError, discount, reset):
        VI = ValueIteration(self.car, self.track, self.memory, reset)
        valuesTable = VI.value_iteration(bellmanError, discount)    # run Value Iteration
        valuesTable.to_csv('./values_tables/R_tables/values_table_RR_' + str(bellmanError) + '_' + str(discount) + '.csv') # save values table to csv file
        print("done " + str(bellmanError) + ' ' + str(discount))
        VI.bestStart() # find best policy/path


    def qLearning(self, filename):        
        QL = QLearning(self.car, self.track, 0.1, 0.9, 0.3, filename, self.memory)
        QL.q_learning()


    def sarsa(self, filename):
        S = SARSA(self.car, self.track, 0.1, 0.9, 0.3, filename, self.memory)
        S.sarsa()
