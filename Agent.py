import numpy as np
import pandas as pd
import time
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
        valuesTable.to_csv('./values_tables/O_tables/values_table_O_' + str(bellmanError) + '_' + str(discount) + '.csv') # save values table to csv file
        print("done " + str(bellmanError) + ' ' + str(discount))
        VI.bestStart() # find best policy/path


    def qLearning(self, filename):        
        QL = QLearning(self.car, self.track, 0.1, 0.9, 0.3, filename, self.memory)
        QL.q_learning()


    def sarsa(self, filename):
        S = SARSA(self.car, self.track, 0.1, 0.9, 0.3, filename, self.memory)
        S.sarsa()

    def drive(self, trackName):
        self.memory['Unnamed: 0'] = self.memory['Unnamed: 0'].apply(eval)
        self.memory = self.memory.set_index("Unnamed: 0")
        startStates = self.getStartStates()
        maxValues = self.getMaxValues(startStates)

        startState = startStates[maxValues.index(max(maxValues))]
        self.car.updatePosition(startState[0], startState[1])
        self.car.updateVelocity(0, 0)

        moves = 0
        finished = False
        while not finished:
            pos = self.car.getPosition()
            originalValue = self.track.track[pos[0]][pos[1]]
            self.track.track[pos[0]][pos[1]] = "C"
            # print(self.track)

            state = self.getState()
            action = self.getAction(state)
            finished = self.takeAction(state, action, trackName)
            

            self.track.track[pos[0]][pos[1]] = originalValue
            moves += 1
            # time.sleep(0.5)

        pos = self.car.getPosition()
        self.track.track[pos[0]][pos[1]] = "C"
        # print(self.track)
        return moves

    def takeAction(self, state, action, trackName):
        action = eval(action)
        # actions fails with a probability of 0.2
        if np.random.rand() <= 0.2:
            action = (0, 0)

        # takes the acceleration action on our car and move
        self.car.updateAcceleration(action[0], action[1])
        self.car.calcVelocity()
        # moveCar returns True or False depending on whether or not a finish
        # line was reached
        finished = self.moveCar(trackName)

        return finished

    def moveCar(self, trackName):

        # finds the cars target position and creates a list of cells the car
        # will pass through
        currentPos = self.car.getPosition()
        velocity = self.car.getVelocity()
        newPos = [currentPos[0] + velocity[0], currentPos[1] + velocity[1]]
        path = self.track.detectWall(currentPos[0], currentPos[1], newPos[0], newPos[1])

        # moves through the path one cell at a time stopping if the car hits
        # a wall or reaches the finish line
        previous = currentPos
        for pos in path:
            # return True if the car finishes
            if self.track.getCell(pos[0], pos[1]) == "F":
                self.car.updatePosition(pos[0], pos[1])
                return True
                
            # if a wall is hit, update position and velocity
            if self.track.getCell(pos[0], pos[1]) == "#" and (trackName == "QL_R_Hard.csv" or trackName == "SARSA_R_Hard.csv"):
                startStates = self.getStartStates()
                maxValues = self.getMaxValues(startStates)

                start = startStates[maxValues.index(max(maxValues))]
                self.car.updatePosition(start[0], start[1])
                self.car.updateVelocity(0, 0)
                
            if self.track.getCell(pos[0], pos[1]) == "#":
                self.car.updatePosition(previous[0], previous[1])
                self.car.updateVelocity(0, 0)
                return False

            previous = pos

        # moves the car to its final position
        self.car.updatePosition(previous[0], previous[1])

        return False

    def getAction(self, state):
        action = self.memory.loc[[state]].idxmax(axis=1).values[0]
        return action

    # returns a state tuple
    def getState(self):
        pos = self.car.getPosition()
        velocity = self.car.getVelocity()
        state = (pos[0], pos[1], velocity[0], velocity[1])
        integerState = tuple(int(value) for value in state)
        return integerState

    def getStartStates(self):
        states = []
        positions = self.track.startPos
        for pos in positions:
            states.append((pos[0], pos[1], 0, 0))

        return states

    def getMaxValues(self, states):
        values = []
        for state in states:
            values.append(max(self.memory.loc[[state]].values[0]))

        return values
