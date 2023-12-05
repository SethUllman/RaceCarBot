import pandas as pd
from Agent import Agent

# qlearner = Agent("./tracks/L-Track/L-track-2.txt")
# QTable = qlearner.qLearning()


def tuneVI():
    bellmanErrors = [0.1, 0.2, 0.3, 0.4, 0.5]
    discounts = [0.8, 0.85, 0.9, 0.95, 1.0]

    for bE in bellmanErrors:
        for d in discounts:
            vIter = Agent("./tracks/R-Track/R-track-full.txt")
            values = vIter.valueIteration(bE, d)

tuneVI()









