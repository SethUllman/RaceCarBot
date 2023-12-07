import pandas as pd
from Agent import Agent

# print("Training L")
# qlearner = Agent("./tracks/L-Track/L-track-full.txt")
# QTable = qlearner.qLearning()
# QTable.to_csv("QL_L.csv")

# print("Training O")
# qlearner = Agent("./tracks/O-Track/O-track-full.txt")
# QTable = qlearner.qLearning()
# QTable.to_csv("QL_O.csv")

# print("Training R")
# qlearner = Agent("./tracks/O-Track/R-track-full.txt")
# QTable = qlearner.qLearning()
# QTable.to_csv("QL_R.csv")

# print("Training O")
# qlearner = Agent("./tracks/O-Track/W-track-full.txt")
# QTable = qlearner.qLearning()
# QTable.to_csv("QL_W.csv")

print("Training L")
sarsaLearner = Agent("./tracks/L-Track/L-track-full.txt")
QTable = sarsaLearner.sarsa()
QTable.to_csv("SARSA_L.csv")

print("Training O")
sarsaLearner = Agent("./tracks/O-Track/L-track-full.txt")
QTable = sarsaLearner.sarsa()
QTable.to_csv("SARSA_O.csv")

print("Training R")
sarsaLearner = Agent("./tracks/R-Track/L-track-full.txt")
QTable = sarsaLearner.sarsa()
QTable.to_csv("SARSA_R.csv")

print("Training O")
sarsaLearner = Agent("./tracks/W-Track/L-track-full.txt")
QTable = sarsaLearner.sarsa()
QTable.to_csv("SARSA_W.csv")

# def tuneVI():
#     bellmanErrors = [0.1, 0.2, 0.3, 0.4, 0.5]
#     discounts = [0.8, 0.85, 0.9, 0.95, 1.0]

#     for bE in bellmanErrors:
#         for d in discounts:
#             vIter = Agent("./tracks/W-Track/W-track-full.txt")
#             values = vIter.valueIteration(bE, d)

# tuneVI()
