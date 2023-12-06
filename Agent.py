import numpy as np
import pandas as pd
from Car import Car
from Track import Track
from QLearning import QLearning

# ---------------- Agent ----------------
class Agent:

    def __init__(self, filename):
        self.car = Car()
        self.track = Track()
        self.track.parseTrack(filename)
        self.memory = pd.DataFrame()

    def valueIteration(self):
        VI = ValueIteration(self.car, self.track, self.memory)

    def qLearning(self):
        QL = QLearning(self.car, self.track, 0.1, 0.9, 0.5)
        qtable = QL.q_learning()
        qtable.to_csv('q_table_full5.csv')
        
        QL = QLearning(self.car, self.track, 0.1, 0.9, 0.4)
        qtable = QL.q_learning()
        qtable.to_csv('q_table_full4.csv')

        QL = QLearning(self.car, self.track, 0.1, 0.9, 0.3)
        qtable = QL.q_learning()
        qtable.to_csv('q_table_full3.csv')

        QL = QLearning(self.car, self.track, 0.1, 0.9, 0.2)
        qtable = QL.q_learning()
        qtable.to_csv('q_table_full2.csv')

        QL = QLearning(self.car, self.track, 0.1, 0.9, 0.1)
        qtable = QL.q_learning()
        qtable.to_csv('q_table_full1.csv')


    def sarsa(self):
        S = SARSA(self.car, self.track, self.memory)