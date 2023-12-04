import numpy as np
import pandas as pd
import random

class QLearning:
  def __init__(self, car, track, memory = None):
    self.actionSpace = 9
    self.stateSpace = track.stateSpace * 11 * 11
    self.car = car
    self.track = track
    if memory is not None and not memory.empty:
      self.Q = memory     #Q-Table
    else:
      indexTuples = [(row, col, x, y) for row in range(0, self.track.row) for col in range(0, self.track.col) for x in range(-5, 6) for y in range(-5, 6)]
      labels = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
      self.Q = pd.DataFrame(index=indexTuples, columns=labels, dtype=float)
      self.Q.fillna(0, inplace=True)
    self.alpha = 0.1    #learning rate
    self.y = 0.9        #discount rate
    self.epsilon = 0.5        #exploration rate, commonly 0.1-0.5
    

  def q_learning(self):
    self.car.updatePosition(self.track.startPos[0][0], self.track.startPos[0][1])
    episodes = 100
    for episode in range(episodes):
      
      state = self.getState(self.car.getPosition())
      totalReward = 0
      for step in range(500):
        action = self.getAction(state)
        nextState, reward = self.takeAction(state, action)
        bestNextAction = self.Q.loc[[nextState]].idxmax(axis=1).values[0]
        currentQValue = self.Q.loc[[state], [action]].values[0][0]
        nextQValue = self.Q.loc[[nextState], [bestNextAction]].values[0][0]
        newQValue = (1 - self.alpha) * currentQValue + self.alpha * (reward + self.y * nextQValue)
        
        self.Q.loc[[state], [action]] = newQValue

    return self.Q

  def getState(self, pos):
    velocity = self.car.getVelocity()
    state = (pos[0], pos[1], velocity[0], velocity[1])
    integerState = tuple(int(value) for value in state)
    return integerState

  def getAction(self, state):
    # Exploration-exploitation trade-off
    if np.random.rand() < self.epsilon:
      # Explore: Choose a random action
      action = self.getRandAction()
    else:
      # Exploit: Choose the action with the highest Q-value for the current state
      action = self.Q.loc[[state]].idxmax(axis=1).values[0]

    return action

  def getRandAction(self):
    acceleration = self.car.getAcceleration()
    new_x, new_y = 0, 0
    possibleValues = [-1, 0, 1]
    if acceleration[0] == 5:
      new_x = possibleValues[random.randrange(0,2)]
    elif acceleration[0] == -5:
      new_x = possibleValues[random.randrange(1,3)]
    else:
      new_x = possibleValues[random.randrange(0,3)]

    if acceleration[1] == 5:
      new_y = possibleValues[random.randrange(0,2)]
    elif acceleration[1] == -5:
      new_y = possibleValues[random.randrange(1,3)]
    else:
      new_y = possibleValues[random.randrange(0,3)]

    return (new_x, new_y)
  
  def takeAction(self, state, action):
    if np.random.rand() <= 0.2:
      action = (0, 0)
    self.car.updateAcceleration(action[0], action[1])
    self.car.calcVelocity()
    finished = self.moveCar()
    reward = 0
    if finished == 0:
      reward = 1
    else: reward = -1

    pos = self.car.getPosition()
    velocity = self.car.getVelocity()
    newState = (pos[0], pos[1], velocity[0], velocity[1])
    return newState, reward

  def moveCar(self):
    currentPos = self.car.getPosition()
    velocity = self.car.getVelocity()
    newPos = [currentPos[0] + velocity[0], currentPos[1] + velocity[1]]
    path = self.track.detectWall(currentPos[0], currentPos[1], newPos[0], newPos[1])

    previous = currentPos
    for pos in path:
      if self.track.getCell(pos[0], pos[1]) == "F":
        return True
        
      if self.track.getCell(pos[0], pos[1]) == "#":
        self.car.updatePosition(previous[0], previous[1])
        self.car.updateVelocity(0, 0)
        return False

    return False

  



    


