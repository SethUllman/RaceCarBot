import pandas as pd
import ast
from Agent import Agent

# while True:
#   # Train Track L
#   print("Training L-1")
#   qlearner = Agent("./tracks/L-Track/L-track-1.txt", "./QLearningTables/QL_L.csv")
#   qlearner.qLearning("QL_L.csv")

#   print("Training L-2")
#   qlearner = Agent("./tracks/L-Track/L-track-2.txt", "./QLearningTables/QL_L.csv")
#   qlearner.qLearning("QL_L.csv")

#   print("Training L-3")
#   qlearner = Agent("./tracks/L-Track/L-track-3.txt", "./QLearningTables/QL_L.csv")
#   qlearner.qLearning("QL_L.csv")

#   print("Training L-4")
#   qlearner = Agent("./tracks/L-Track/L-track-4.txt", "./QLearningTables/QL_L.csv")
#   qlearner.qLearning("QL_L.csv")

#   print("Training L-5")
#   qlearner = Agent("./tracks/L-Track/L-track-5.txt", "./QLearningTables/QL_L.csv")
#   qlearner.qLearning("QL_L.csv")

#   print("Training L-full")
#   qlearner = Agent("./tracks/L-Track/L-track-full.txt", "./QLearningTables/QL_L.csv")
#   qlearner.qLearning("QL_L.csv")

#   # Train Track O
#   print("Training O-1")
#   qlearner = Agent("./tracks/O-Track/O-track-1.txt", "./QLearningTables/QL_O.csv")
#   qlearner.qLearning("QL_O.csv")

#   print("Training O-2")
#   qlearner = Agent("./tracks/O-Track/O-track-2.txt", "./QLearningTables/QL_O.csv")
#   qlearner.qLearning("QL_O.csv")

#   print("Training O-3")
#   qlearner = Agent("./tracks/O-Track/O-track-3.txt", "./QLearningTables/QL_O.csv")
#   qlearner.qLearning("QL_O.csv")

#   print("Training O-4")
#   qlearner = Agent("./tracks/O-Track/O-track-4.txt", "./QLearningTables/QL_O.csv")
#   qlearner.qLearning("QL_O.csv")

#   print("Training O-5")
#   qlearner = Agent("./tracks/O-Track/O-track-5.txt", "./QLearningTables/QL_O.csv")
#   qlearner.qLearning("QL_O.csv")

#   print("Training O-6")
#   qlearner = Agent("./tracks/O-Track/O-track-6.txt", "./QLearningTables/QL_O.csv")
#   qlearner.qLearning("QL_O.csv")

#   print("Training O-7")
#   qlearner = Agent("./tracks/O-Track/O-track-7.txt", "./QLearningTables/QL_O.csv")
#   qlearner.qLearning("QL_O.csv")

#   print("Training O-8")
#   qlearner = Agent("./tracks/O-Track/O-track-8.txt", "./QLearningTables/QL_O.csv")
#   qlearner.qLearning("QL_O.csv")

#   print("Training O-9")
#   qlearner = Agent("./tracks/O-Track/O-track-9.txt", "./QLearningTables/QL_O.csv")
#   qlearner.qLearning("QL_O.csv")

#   print("Training O-full")
#   qlearner = Agent("./tracks/O-Track/O-track-full.txt", "./QLearningTables/QL_O.csv")
#   qlearner.qLearning("QL_O.csv")

#   # Train Track R
#   print("Training R-1")
#   qlearner = Agent("./tracks/R-Track/R-track-1.txt", "./QLearningTables/QL_R.csv")
#   qlearner.qLearning("QL_R.csv")

#   print("Training R-2")
#   qlearner = Agent("./tracks/R-Track/R-track-2.txt", "./QLearningTables/QL_R.csv")
#   qlearner.qLearning("QL_R.csv")

#   print("Training R-3")
#   qlearner = Agent("./tracks/R-Track/R-track-3.txt", "./QLearningTables/QL_R.csv")
#   qlearner.qLearning("QL_R.csv")

#   print("Training R-4")
#   qlearner = Agent("./tracks/R-Track/R-track-4.txt", "./QLearningTables/QL_R.csv")
#   qlearner.qLearning("QL_R.csv")

#   print("Training R-5")
#   qlearner = Agent("./tracks/R-Track/R-track-5.txt", "./QLearningTables/QL_R.csv")
#   qlearner.qLearning("QL_R.csv")

#   print("Training R-6")
#   qlearner = Agent("./tracks/R-Track/R-track-6.txt", "./QLearningTables/QL_R.csv")
#   qlearner.qLearning("QL_R.csv")

#   print("Training R-7")
#   qlearner = Agent("./tracks/R-Track/R-track-7.txt", "./QLearningTables/QL_R.csv")
#   qlearner.qLearning("QL_R.csv")

#   print("Training R-8")
#   qlearner = Agent("./tracks/R-Track/R-track-8.txt", "./QLearningTables/QL_R.csv")
#   qlearner.qLearning("QL_R.csv")

#   print("Training R-9")
#   qlearner = Agent("./tracks/R-Track/R-track-9.txt", "./QLearningTables/QL_R.csv")
#   qlearner.qLearning("QL_R.csv")

#   print("Training R-10")
#   qlearner = Agent("./tracks/R-Track/R-track-10.txt", "./QLearningTables/QL_R.csv")
#   qlearner.qLearning("QL_R.csv")

#   print("Training R-full")
#   qlearner = Agent("./tracks/R-Track/R-track-full.txt", "./QLearningTables/QL_R.csv")
#   qlearner.qLearning("QL_R.csv")

#   # Train Track R
#   print("Training R-1 Hard")
#   qlearner = Agent("./tracks/R-Track/R-track-1.txt", "./QLearningTables/QL_R_Hard.csv")
#   qlearner.qLearning("QL_R_Hard.csv")

#   print("Training R-2 Hard")
#   qlearner = Agent("./tracks/R-Track/R-track-2.txt", "./QLearningTables/QL_R_Hard.csv")
#   qlearner.qLearning("QL_R_Hard.csv")

#   print("Training R-3 Hard")
#   qlearner = Agent("./tracks/R-Track/R-track-3.txt", "./QLearningTables/QL_R_Hard.csv")
#   qlearner.qLearning("QL_R_Hard.csv")

#   print("Training R-4 Hard")
#   qlearner = Agent("./tracks/R-Track/R-track-4.txt", "./QLearningTables/QL_R_Hard.csv")
#   qlearner.qLearning("QL_R_Hard.csv")

#   print("Training R-5 Hard")
#   qlearner = Agent("./tracks/R-Track/R-track-5.txt", "./QLearningTables/QL_R_Hard.csv")
#   qlearner.qLearning("QL_R_Hard.csv")

#   print("Training R-6 Hard")
#   qlearner = Agent("./tracks/R-Track/R-track-6.txt", "./QLearningTables/QL_R_Hard.csv")
#   qlearner.qLearning("QL_R_Hard.csv")

#   print("Training R-7 Hard")
#   qlearner = Agent("./tracks/R-Track/R-track-7.txt", "./QLearningTables/QL_R_Hard.csv")
#   qlearner.qLearning("QL_R_Hard.csv")

#   print("Training R-8 Hard")
#   qlearner = Agent("./tracks/R-Track/R-track-8.txt", "./QLearningTables/QL_R_Hard.csv")
#   qlearner.qLearning("QL_R_Hard.csv")

#   print("Training R-9 Hard")
#   qlearner = Agent("./tracks/R-Track/R-track-9.txt", "./QLearningTables/QL_R_Hard.csv")
#   qlearner.qLearning("QL_R_Hard.csv")

#   print("Training R-10 Hard")
#   qlearner = Agent("./tracks/R-Track/R-track-10.txt", "./QLearningTables/QL_R_Hard.csv")
#   qlearner.qLearning("QL_R_Hard.csv")

#   print("Training R-full Hard")
#   qlearner = Agent("./tracks/R-Track/R-track-full.txt", "./QLearningTables/QL_R_Hard.csv")
#   qlearner.qLearning("QL_R_Hard.csv")

#   # Train Track W
#   print("Training W-1")
#   qlearner = Agent("./tracks/W-Track/W-track-1.txt", "./QLearningTables/QL_W.csv")
#   qlearner.qLearning("QL_W.csv")

#   print("Training W-2")
#   qlearner = Agent("./tracks/W-Track/W-track-2.txt", "./QLearningTables/QL_W.csv")
#   qlearner.qLearning("QL_W.csv")

# print("Training L")
# sarsaLearner = Agent("./tracks/L-Track/L-track-full.txt")
# QTable = sarsaLearner.sarsa("SARSA_L.csv")

# print("Training O")
# sarsaLearner = Agent("./tracks/O-Track/O-track-full.txt")
# QTable = sarsaLearner.sarsa("SARSA_O.csv")

# print("Training R")
# sarsaLearner = Agent("./tracks/R-Track/R-track-full.txt")
# QTable = sarsaLearner.sarsa("SARSA_R.csv")

# print("Training O")
# sarsaLearner = Agent("./tracks/W-Track/W-track-full.txt")
# QTable = sarsaLearner.sarsa("SARSA_W.csv")

# tune Bellman Error and Discount Factor for Value Iteration
def tuneVI():
    reset = True
    bellmanErrors = [0.1, 0.15, 0.2, 0.25, 0.3]
    discounts = [0.8, 0.85, 0.9, 0.95, 1.0]

    for bE in bellmanErrors:
        for d in discounts:
            vIter = Agent("./tracks/R-Track/R-track-full.txt")
            values = vIter.valueIteration(bE, d, reset)

tuneVI()

# def costResults():
#     track = "W"
#     reset = False
#     bellmanErrors = [0.1, 0.2, 0.3, 0.4, 0.5]
#     discounts = [0.8, 0.85, 0.9, 0.95, 1.0]
#     vIter = Agent('./tracks/' + track + '-Track/' + track + '-track-full.txt')

#     for bE in bellmanErrors:
#         for d in discounts:
#             df_VI = pd.read_csv('./values_tables/' + track + '_tables/values_table_' + track + '_' + str(bE) + '_' + str(d) + '.csv')
#             print(track + "-Track Best Policy with Bellman Error=" + str(bE) + " & Discount Factor=" + str(d))
#             vIter.memory = df_VI
#             vIter.valueIteration(bE, d, reset)

# tuneVI()
