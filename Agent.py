import numpy as np
import pandas as pd
from Car import Car
from Track import Track
from ValueIteration import ValueIteration
from QLearning import QLearning

# ---------------- Agent ----------------
class Agent:

    def __init__(self, filename):
        self.car = Car()
        self.track = Track()
        self.track.parseTrack(filename)
        self.memory = pd.DataFrame()

    def valueIteration(self, bellmanError, discount):
        VI = ValueIteration(self.car, self.track, self.memory)
        valuesTable = VI.value_iteration(bellmanError, discount)
        valuesTable.to_csv('./values_tables/R_tables/values_table_RC_' + str(bellmanError) + '_' + str(discount) + '.csv')
        print("done " + str(bellmanError) + ' ' +str(discount))


    def qLearning(self):
        QL = QLearning(self.car, self.track, self.memory)
        qtable = QL.q_learning()
        qtable.to_csv('q_table_L1.csv')


    def sarsa(self):
        S = SARSA(self.car, self.track, self.memory)