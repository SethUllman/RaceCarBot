import numpy as np
import pandas as pd
import Car
import Track

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
        QL = QLearning(self.car, self.track, self.memory)

    def sarsa(self):
        S = SARSA(self.car, self.track, self.memory)