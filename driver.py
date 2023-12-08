import pandas as pd
from Agent import Agent

while True:
  # Train Track L
  print("Training L-1")
  qlearner = Agent("./tracks/L-Track/L-track-1.txt", "./QLearningTables/QL_L.csv")
  qlearner.qLearning("QL_L.csv")

  print("Training L-2")
  qlearner = Agent("./tracks/L-Track/L-track-2.txt", "./QLearningTables/QL_L.csv")
  qlearner.qLearning("QL_L.csv")

  print("Training L-3")
  qlearner = Agent("./tracks/L-Track/L-track-3.txt", "./QLearningTables/QL_L.csv")
  qlearner.qLearning("QL_L.csv")

  print("Training L-4")
  qlearner = Agent("./tracks/L-Track/L-track-4.txt", "./QLearningTables/QL_L.csv")
  qlearner.qLearning("QL_L.csv")

  print("Training L-5")
  qlearner = Agent("./tracks/L-Track/L-track-5.txt", "./QLearningTables/QL_L.csv")
  qlearner.qLearning("QL_L.csv")

  print("Training L-full")
  qlearner = Agent("./tracks/L-Track/L-track-full.txt", "./QLearningTables/QL_L.csv")
  qlearner.qLearning("QL_L.csv")

  # Train Track O
  print("Training O-1")
  qlearner = Agent("./tracks/O-Track/O-track-1.txt", "./QLearningTables/QL_O.csv")
  qlearner.qLearning("QL_O.csv")

  print("Training O-2")
  qlearner = Agent("./tracks/O-Track/O-track-2.txt", "./QLearningTables/QL_O.csv")
  qlearner.qLearning("QL_O.csv")

  print("Training O-3")
  qlearner = Agent("./tracks/O-Track/O-track-3.txt", "./QLearningTables/QL_O.csv")
  qlearner.qLearning("QL_O.csv")

  print("Training O-4")
  qlearner = Agent("./tracks/O-Track/O-track-4.txt", "./QLearningTables/QL_O.csv")
  qlearner.qLearning("QL_O.csv")

  print("Training O-5")
  qlearner = Agent("./tracks/O-Track/O-track-5.txt", "./QLearningTables/QL_O.csv")
  qlearner.qLearning("QL_O.csv")

  print("Training O-6")
  qlearner = Agent("./tracks/O-Track/O-track-6.txt", "./QLearningTables/QL_O.csv")
  qlearner.qLearning("QL_O.csv")

  print("Training O-7")
  qlearner = Agent("./tracks/O-Track/O-track-7.txt", "./QLearningTables/QL_O.csv")
  qlearner.qLearning("QL_O.csv")

  print("Training O-8")
  qlearner = Agent("./tracks/O-Track/O-track-8.txt", "./QLearningTables/QL_O.csv")
  qlearner.qLearning("QL_O.csv")

  print("Training O-9")
  qlearner = Agent("./tracks/O-Track/O-track-9.txt", "./QLearningTables/QL_O.csv")
  qlearner.qLearning("QL_O.csv")

  print("Training O-full")
  qlearner = Agent("./tracks/O-Track/O-track-full.txt", "./QLearningTables/QL_O.csv")
  qlearner.qLearning("QL_O.csv")

  # Train Track R
  print("Training R-1")
  qlearner = Agent("./tracks/R-Track/R-track-1.txt", "./QLearningTables/QL_R.csv")
  qlearner.qLearning("QL_R.csv")

  print("Training R-2")
  qlearner = Agent("./tracks/R-Track/R-track-2.txt", "./QLearningTables/QL_R.csv")
  qlearner.qLearning("QL_R.csv")

  print("Training R-3")
  qlearner = Agent("./tracks/R-Track/R-track-3.txt", "./QLearningTables/QL_R.csv")
  qlearner.qLearning("QL_R.csv")

  print("Training R-4")
  qlearner = Agent("./tracks/R-Track/R-track-4.txt", "./QLearningTables/QL_R.csv")
  qlearner.qLearning("QL_R.csv")

  print("Training R-5")
  qlearner = Agent("./tracks/R-Track/R-track-5.txt", "./QLearningTables/QL_R.csv")
  qlearner.qLearning("QL_R.csv")

  print("Training R-6")
  qlearner = Agent("./tracks/R-Track/R-track-6.txt", "./QLearningTables/QL_R.csv")
  qlearner.qLearning("QL_R.csv")

  print("Training R-7")
  qlearner = Agent("./tracks/R-Track/R-track-7.txt", "./QLearningTables/QL_R.csv")
  qlearner.qLearning("QL_R.csv")

  print("Training R-8")
  qlearner = Agent("./tracks/R-Track/R-track-8.txt", "./QLearningTables/QL_R.csv")
  qlearner.qLearning("QL_R.csv")

  print("Training R-9")
  qlearner = Agent("./tracks/R-Track/R-track-9.txt", "./QLearningTables/QL_R.csv")
  qlearner.qLearning("QL_R.csv")

  print("Training R-10")
  qlearner = Agent("./tracks/R-Track/R-track-10.txt", "./QLearningTables/QL_R.csv")
  qlearner.qLearning("QL_R.csv")

  print("Training R-full")
  qlearner = Agent("./tracks/R-Track/R-track-full.txt", "./QLearningTables/QL_R.csv")
  qlearner.qLearning("QL_R.csv")

  # Train Track R
  print("Training R-1 Hard")
  qlearner = Agent("./tracks/R-Track/R-track-1.txt", "./QLearningTables/QL_R_Hard.csv")
  qlearner.qLearning("QL_R_Hard.csv")

  print("Training R-2 Hard")
  qlearner = Agent("./tracks/R-Track/R-track-2.txt", "./QLearningTables/QL_R_Hard.csv")
  qlearner.qLearning("QL_R_Hard.csv")

  print("Training R-3 Hard")
  qlearner = Agent("./tracks/R-Track/R-track-3.txt", "./QLearningTables/QL_R_Hard.csv")
  qlearner.qLearning("QL_R_Hard.csv")

  print("Training R-4 Hard")
  qlearner = Agent("./tracks/R-Track/R-track-4.txt", "./QLearningTables/QL_R_Hard.csv")
  qlearner.qLearning("QL_R_Hard.csv")

  print("Training R-5 Hard")
  qlearner = Agent("./tracks/R-Track/R-track-5.txt", "./QLearningTables/QL_R_Hard.csv")
  qlearner.qLearning("QL_R_Hard.csv")

  print("Training R-6 Hard")
  qlearner = Agent("./tracks/R-Track/R-track-6.txt", "./QLearningTables/QL_R_Hard.csv")
  qlearner.qLearning("QL_R_Hard.csv")

  print("Training R-7 Hard")
  qlearner = Agent("./tracks/R-Track/R-track-7.txt", "./QLearningTables/QL_R_Hard.csv")
  qlearner.qLearning("QL_R_Hard.csv")

  print("Training R-8 Hard")
  qlearner = Agent("./tracks/R-Track/R-track-8.txt", "./QLearningTables/QL_R_Hard.csv")
  qlearner.qLearning("QL_R_Hard.csv")

  print("Training R-9 Hard")
  qlearner = Agent("./tracks/R-Track/R-track-9.txt", "./QLearningTables/QL_R_Hard.csv")
  qlearner.qLearning("QL_R_Hard.csv")

  print("Training R-10 Hard")
  qlearner = Agent("./tracks/R-Track/R-track-10.txt", "./QLearningTables/QL_R_Hard.csv")
  qlearner.qLearning("QL_R_Hard.csv")

  print("Training R-full Hard")
  qlearner = Agent("./tracks/R-Track/R-track-full.txt", "./QLearningTables/QL_R_Hard.csv")
  qlearner.qLearning("QL_R_Hard.csv")

  # Train Track W
  print("Training W-1")
  qlearner = Agent("./tracks/W-Track/W-track-1.txt", "./QLearningTables/QL_W.csv")
  qlearner.qLearning("QL_W.csv")

  print("Training W-2")
  qlearner = Agent("./tracks/W-Track/W-track-2.txt", "./QLearningTables/QL_W.csv")
  qlearner.qLearning("QL_W.csv")

  print("Training W-3")
  qlearner = Agent("./tracks/W-Track/W-track-3.txt", "./QLearningTables/QL_W.csv")
  qlearner.qLearning("QL_W.csv")

  print("Training W-4")
  qlearner = Agent("./tracks/W-Track/W-track-4.txt", "./QLearningTables/QL_W.csv")
  qlearner.qLearning("QL_W.csv")

  print("Training W-full")
  qlearner = Agent("./tracks/W-Track/W-track-full.txt", "./QLearningTables/QL_W.csv")
  qlearner.qLearning("QL_W.csv")

  #--------------------------SARSA-------------------------

  # Train Track L
  print("Training L-1")
  sarsaLearner = Agent("./tracks/L-Track/L-track-1.txt", "./SarsaTables/SARSA_L.csv")
  sarsaLearner.sarsa("SARSA_L.csv")

  print("Training L-2")
  sarsaLearner = Agent("./tracks/L-Track/L-track-2.txt", "./SarsaTables/SARSA_L.csv")
  sarsaLearner.sarsa("SARSA_L.csv")

  print("Training L-3")
  sarsaLearner = Agent("./tracks/L-Track/L-track-3.txt", "./SarsaTables/SARSA_L.csv")
  sarsaLearner.sarsa("SARSA_L.csv")

  print("Training L-4")
  sarsaLearner = Agent("./tracks/L-Track/L-track-4.txt", "./SarsaTables/SARSA_L.csv")
  sarsaLearner.sarsa("SARSA_L.csv")

  print("Training L-5")
  sarsaLearner = Agent("./tracks/L-Track/L-track-5.txt", "./SarsaTables/SARSA_L.csv")
  sarsaLearner.sarsa("SARSA_L.csv")

  print("Training L-full")
  sarsaLearner = Agent("./tracks/L-Track/L-track-full.txt", "./SarsaTables/SARSA_L.csv")
  sarsaLearner.sarsa("SARSA_L.csv")

  # Train Track O
  print("Training O-1")
  sarsaLearner = Agent("./tracks/O-Track/O-track-1.txt", "./SarsaTables/SARSA_O.csv")
  sarsaLearner.sarsa("SARSA_O.csv")

  print("Training O-2")
  sarsaLearner = Agent("./tracks/O-Track/O-track-2.txt", "./SarsaTables/SARSA_O.csv")
  sarsaLearner.sarsa("SARSA_O.csv")

  print("Training O-3")
  sarsaLearner = Agent("./tracks/O-Track/O-track-3.txt", "./SarsaTables/SARSA_O.csv")
  sarsaLearner.sarsa("SARSA_O.csv")

  print("Training O-4")
  sarsaLearner = Agent("./tracks/O-Track/O-track-4.txt", "./SarsaTables/SARSA_O.csv")
  sarsaLearner.sarsa("SARSA_O.csv")

  print("Training O-5")
  sarsaLearner = Agent("./tracks/O-Track/O-track-5.txt", "./SarsaTables/SARSA_O.csv")
  sarsaLearner.sarsa("SARSA_O.csv")

  print("Training O-6")
  sarsaLearner = Agent("./tracks/O-Track/O-track-6.txt", "./SarsaTables/SARSA_O.csv")
  sarsaLearner.sarsa("SARSA_O.csv")

  print("Training O-7")
  sarsaLearner = Agent("./tracks/O-Track/O-track-7.txt", "./SarsaTables/SARSA_O.csv")
  sarsaLearner.sarsa("SARSA_O.csv")

  print("Training O-8")
  sarsaLearner = Agent("./tracks/O-Track/O-track-8.txt", "./SarsaTables/SARSA_O.csv")
  sarsaLearner.sarsa("SARSA_O.csv")

  print("Training O-9")
  sarsaLearner = Agent("./tracks/O-Track/O-track-9.txt", "./SarsaTables/SARSA_O.csv")
  sarsaLearner.sarsa("SARSA_O.csv")

  print("Training O-full")
  sarsaLearner = Agent("./tracks/O-Track/O-track-full.txt", "./SarsaTables/SARSA_O.csv")
  sarsaLearner.sarsa("SARSA_O.csv")

  # Train Track R
  print("Training R-1")
  sarsaLearner = Agent("./tracks/R-Track/R-track-1.txt", "./SarsaTables/SARSA_R.csv")
  sarsaLearner.sarsa("SARSA_R.csv")

  print("Training R-2")
  sarsaLearner = Agent("./tracks/R-Track/R-track-2.txt", "./SarsaTables/SARSA_R.csv")
  sarsaLearner.sarsa("SARSA_R.csv")

  print("Training R-3")
  sarsaLearner = Agent("./tracks/R-Track/R-track-3.txt", "./SarsaTables/SARSA_R.csv")
  sarsaLearner.sarsa("SARSA_R.csv")

  print("Training R-4")
  sarsaLearner = Agent("./tracks/R-Track/R-track-4.txt", "./SarsaTables/SARSA_R.csv")
  sarsaLearner.sarsa("SARSA_R.csv")

  print("Training R-5")
  sarsaLearner = Agent("./tracks/R-Track/R-track-5.txt", "./SarsaTables/SARSA_R.csv")
  sarsaLearner.sarsa("SARSA_R.csv")

  print("Training R-6")
  sarsaLearner = Agent("./tracks/R-Track/R-track-6.txt", "./SarsaTables/SARSA_R.csv")
  sarsaLearner.sarsa("SARSA_R.csv")

  print("Training R-7")
  sarsaLearner = Agent("./tracks/R-Track/R-track-7.txt", "./SarsaTables/SARSA_R.csv")
  sarsaLearner.sarsa("SARSA_R.csv")

  print("Training R-8")
  sarsaLearner = Agent("./tracks/R-Track/R-track-8.txt", "./SarsaTables/SARSA_R.csv")
  sarsaLearner.sarsa("SARSA_R.csv")

  print("Training R-9")
  sarsaLearner = Agent("./tracks/R-Track/R-track-9.txt", "./SarsaTables/SARSA_R.csv")
  sarsaLearner.sarsa("SARSA_R.csv")

  print("Training R-10")
  sarsaLearner = Agent("./tracks/R-Track/R-track-10.txt", "./SarsaTables/SARSA_R.csv")
  sarsaLearner.sarsa("SARSA_R.csv")

  print("Training R-full")
  sarsaLearner = Agent("./tracks/R-Track/R-track-full.txt", "./SarsaTables/SARSA_R.csv")
  sarsaLearner.sarsa("SARSA_R.csv")

  # Train Track R
  print("Training R-1 Hard")
  sarsaLearner = Agent("./tracks/R-Track/R-track-1.txt", "./QLearningTables/QL_R_Hard.csv")
  sarsaLearner.sarsa("QL_R_Hard.csv")

  print("Training R-2 Hard")
  sarsaLearner = Agent("./tracks/R-Track/R-track-2.txt", "./QLearningTables/QL_R_Hard.csv")
  sarsaLearner.sarsa("QL_R_Hard.csv")

  print("Training R-3 Hard")
  sarsaLearner = Agent("./tracks/R-Track/R-track-3.txt", "./QLearningTables/QL_R_Hard.csv")
  sarsaLearner.sarsa("QL_R_Hard.csv")

  print("Training R-4 Hard")
  sarsaLearner = Agent("./tracks/R-Track/R-track-4.txt", "./QLearningTables/QL_R_Hard.csv")
  sarsaLearner.sarsa("QL_R_Hard.csv")

  print("Training R-5 Hard")
  sarsaLearner = Agent("./tracks/R-Track/R-track-5.txt", "./QLearningTables/QL_R_Hard.csv")
  sarsaLearner.sarsa("QL_R_Hard.csv")

  print("Training R-6 Hard")
  sarsaLearner = Agent("./tracks/R-Track/R-track-6.txt", "./QLearningTables/QL_R_Hard.csv")
  sarsaLearner.sarsa("QL_R_Hard.csv")

  print("Training R-7 Hard")
  sarsaLearner = Agent("./tracks/R-Track/R-track-7.txt", "./QLearningTables/QL_R_Hard.csv")
  sarsaLearner.sarsa("QL_R_Hard.csv")

  print("Training R-8 Hard")
  sarsaLearner = Agent("./tracks/R-Track/R-track-8.txt", "./QLearningTables/QL_R_Hard.csv")
  sarsaLearner.sarsa("QL_R_Hard.csv")

  print("Training R-9 Hard")
  sarsaLearner = Agent("./tracks/R-Track/R-track-9.txt", "./QLearningTables/QL_R_Hard.csv")
  sarsaLearner.sarsa("QL_R_Hard.csv")

  print("Training R-10 Hard")
  sarsaLearner = Agent("./tracks/R-Track/R-track-10.txt", "./QLearningTables/QL_R_Hard.csv")
  sarsaLearner.sarsa("QL_R_Hard.csv")

  print("Training R-full Hard")
  sarsaLearner = Agent("./tracks/R-Track/R-track-full.txt", "./QLearningTables/QL_R_Hard.csv")
  sarsaLearner.sarsa("QL_R_Hard.csv")

  # Train Track W
  print("Training W-1")
  sarsaLearner = Agent("./tracks/W-Track/W-track-1.txt", "./SarsaTables/SARSA_W.csv")
  sarsaLearner.sarsa("SARSA_W.csv")

  print("Training W-2")
  sarsaLearner = Agent("./tracks/W-Track/W-track-2.txt", "./SarsaTables/SARSA_W.csv")
  sarsaLearner.sarsa("SARSA_W.csv")

  print("Training W-3")
  sarsaLearner = Agent("./tracks/W-Track/W-track-3.txt", "./SarsaTables/SARSA_W.csv")
  sarsaLearner.sarsa("SARSA_W.csv")

  print("Training W-4")
  sarsaLearner = Agent("./tracks/W-Track/W-track-4.txt", "./SarsaTables/SARSA_W.csv")
  sarsaLearner.sarsa("SARSA_W.csv")

  print("Training W-full")
  sarsaLearner = Agent("./tracks/W-Track/W-track-full.txt", "./SarsaTables/SARSA_W.csv")
  sarsaLearner.sarsa("SARSA_W.csv")

# def tuneVI():
#     bellmanErrors = [0.1, 0.2, 0.3, 0.4, 0.5]
#     discounts = [0.8, 0.85, 0.9, 0.95, 1.0]

#     for bE in bellmanErrors:
#         for d in discounts:
#             vIter = Agent("./tracks/W-Track/W-track-full.txt")
#             values = vIter.valueIteration(bE, d)

# tuneVI()