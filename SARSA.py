import numpy as np
import pandas as pd
import random

class SARSA:
  def __init__(self, car, track, alpha, y, epsilon, filename, memory = None):
    # initialize starting values
    self.actionSpace = 9
    self.stateSpace = track.stateSpace * 11 * 11
    self.car = car
    self.track = track
    self.filename = filename

    # if a memory DataFrame was provided use that as the Q-Table
    # otherwise created a new DataFrame
    if memory is not None and not memory.empty:
      self.Q = memory     #Q-Table
      self.Q['Unnamed: 0'] = self.Q['Unnamed: 0'].apply(eval)
      self.Q = self.Q.set_index("Unnamed: 0")
    else:
      indexTuples = [(row, col, x, y) for row in range(0, self.track.row) for col in range(0, self.track.col) for x in range(-5, 6) for y in range(-5, 6)]
      labels = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
      self.Q = pd.DataFrame(index=indexTuples, columns=labels, dtype=float)
      self.Q.fillna(0, inplace=True)

    # tunable parameters
    self.alpha = alpha      #learning rate
    self.y = y              #discount rate
    self.epsilon = epsilon  #exploration rate, commonly 0.1-0.5
    
  def getStartState(self):
    return random.choice(self.track.startPos)

  # The sarsa Algorithm, returns a Q-Table
  def sarsa(self):  

    # set the number of iterations
    episodes = 1000
    for episode in range(episodes):
      if(episode % 100 == 0): 
        self.Q.to_csv("./SarsaTables/" + self.filename)
      if(episode % 10 == 0):
        print(str(episode/10) + "/" + str(episodes/10))
      start = self.getStartState()
      self.car.updatePosition(start[0], start[1])
      state = self.getState(self.car.getPosition())
      for step in range(50):
        # creates the current state as a tuple (x, y, xv, yv)
        state = self.getState(self.car.getPosition())
        # choose and perform an action, then find values needed for Q-Table update
        action = self.getAction(state)
        if isinstance(action, str):
          action = eval(action)
        nextState, reward = self.takeAction(state, action)
        bestNextAction = self.Q.loc[[nextState]].idxmax(axis=1).values[0]
        currentQValue = self.Q.loc[[state], [str(action)]].values[0][0]
        nextQValue = self.Q.loc[[nextState], [bestNextAction]].values[0][0]

        # update Q-Table
        newQValue = currentQValue + (self.alpha * (reward + (self.y * nextQValue) - currentQValue))
        self.Q.loc[[state], [str(action)]] = newQValue

    return self.Q

  # returns a state tuple
  def getState(self, pos):
    velocity = self.car.getVelocity()
    state = (pos[0], pos[1], velocity[0], velocity[1])
    integerState = tuple(int(value) for value in state)
    return integerState

  # decides whether to get the best action or a random one using
  # the tunable parameter epsilon
  def getAction(self, state):
    # Exploration-exploitation trade-off
    if np.random.rand() < self.epsilon:
      # Explore: Choose a random action
      action = self.getRandAction()
    else:
      # Exploit: Choose the action with the highest Q-value for the current state
      action = self.Q.loc[[state]].idxmax(axis=1).values[0]

    return action

  def getDriveAction(self, state):
    action = self.Q.loc[[state]].idxmax(axis=1).values[0]
    return action

  # creates a random, valid, action
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
  
  # exacutes the found action
  def takeAction(self, state, action):

    # actions fails with a probability of 0.2
    if np.random.rand() <= 0.2:
      action = (0, 0)
      # print("action failed")

    # takes the acceleration action on our car and move
    self.car.updateAcceleration(action[0], action[1])
    self.car.calcVelocity()
    # moveCar returns True or False depending on whether or not a finish
    # line was reached
    finished = self.moveCar()
    reward = 0
    if finished:
      reward = 1
    else: reward = 0

    # create a new state tuple
    pos = self.car.getPosition()
    velocity = self.car.getVelocity()
    newState = (pos[0], pos[1], velocity[0], velocity[1])
    return newState, reward

  # moves the car using its velocity values
  def moveCar(self):

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
        return True
        
      # if a wall is hit, update position and velocity
      if self.track.getCell(pos[0], pos[1]) == "#" and self.filename == "QL_R_Hard":
        start = random.choice(self.track.startPos)
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

  def drive(self):
    pos = self.track.startPos[0]
    self.car.updatePosition(pos[0], pos[1])
    self.car.updateVelocity(0, 0)

    numMoves = 0

    for i in range(50):
      prevVal = self.track.track[pos[0]][pos[1]]
      self.track.track[pos[0]][pos[1]] = "C"
      print(self.track)
      state = self.getState(pos)
      action = self.getDriveAction(state)
      action = eval(action)
      print("State: " + str(state))
      print(str(self.Q.loc[[state]]))
      print("Action: " + str(action))
      print("Position: " + str(self.car.getPosition()) + " Velocity: " + str(self.car.getVelocity()))
      newState, reward = self.takeAction(state, action)
      self.track.track[pos[0]][pos[1]] = prevVal
      pos = self.car.getPosition()
      numMoves += 1
      if self.track.track[pos[0]][pos[1]] == "F":
        print("finished in " + str(numMoves) + " moves")
      
    

  



    


