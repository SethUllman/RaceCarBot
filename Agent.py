import numpy as np
import pandas as pd
from Car import Car
from Track import Track
from ValueIteration import ValueIteration
from QLearning import QLearning
from SARSA import SARSA

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
        valuesTable.to_csv('./values_tables/W_tables/values_table_W_' + str(bellmanError) + '_' + str(discount) + '.csv')
        print("done " + str(bellmanError) + ' ' +str(discount))


    def qLearning(self, filename):
        # table = pd.read_csv('q_table_L1_min.csv')
        
        QL = QLearning(self.car, self.track, 0.1, 0.9, 0.3, filename)
        # QL.drive()
        qtable = QL.q_learning()
        return qtable



    def sarsa(self, filename):
        S = SARSA(self.car, self.track, 0.1, 0.9, 0.3, filename)
        qtable = S.sarsa()
        return qtable