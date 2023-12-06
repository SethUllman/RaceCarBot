from types import new_class
import numpy as np
import pandas as pd
import random

class ValueIteration:
    def __init__(self, car, track, memory):
        self.car = car
        self.track = track
        if memory is not None and not memory.empty:
            self.values = memory     #Q-Table
        else:
            indexTuples = [(row, col, x, y) for row in range(0, self.track.row) for col in range(0, self.track.col) for x in range(-5, 6) for y in range(-5, 6)]
            labels = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
            self.values = pd.DataFrame(index=indexTuples, columns=labels, dtype=float)
            self.values.fillna(float('-1.0'), inplace=True)

    def setFinishLine(self):
        positions = self.track.finishPos
        labels = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

        for pos in positions:
            for vx in range(-5, 6):
                for vy in range(-5, 6):
                    for a in labels:
                        self.values.at[(pos[0], pos[1], vx, vy), a] = 0.0

    def value_iteration(self, bellmanError, discount):
        delta = float('inf')
        count = 0
        self.setFinishLine()

        while delta > bellmanError:
            delta = 0.0

            states = self.values.index.tolist()
            actions = self.values.columns.tolist()

            for state in states:
            
                if [state[0], state[1]] in self.track.finishPos:
                    continue

                v = self.bestValue(state)
                max_value = float('-inf')

                for action in actions:
                    if self.track.getCell(state[0], state[1]) == '#':
                        self.values.at[state, action] = float('-inf')
                        continue

                    value = self.reward(state, action) + discount*self.bestValue(self.getNextState(state, action))
                    self.values.at[state, action] = round(value,2)
                    
                    if value > max_value:
                        max_value = value

                if self.track.getCell(state[0], state[1]) != '#':
                    V = round(max_value,2)
                    if delta < abs(v - V):
                        delta = round(abs(v - V), 4)

                    # print("Delta:", delta, "\tV: ", V)
            count+=1

            print("Count: ",count)
            self.values.to_csv('./values_tables/W_tables/values_table_W_' + str(bellmanError) + '_' +str(discount) + '.csv')

        return self.values

    def reward(self, state, action):
        # Assume state is a tuple (x, y, vx, vy)
        x, y, vx, vy = state
        
        # Check if the current state is on the finish line
        if [x, y] in self.track.finishPos:
            return 0.0  
        
        # Calculate the next state after taking the action
        next_state = self.getNextState(state, action)
        
        # Check if the next state crosses the finish line
        if [next_state[0], next_state[1]] in self.track.finishPos:
            return 0.0  
    
        # Assign a cost of 1 for each move
        return -1.0
        
    def getNextState(self, state, action):
        self.car.updatePosition(state[0], state[1])
        self.car.updateVelocity(state[2], state[3])
        self.car.updateAcceleration(action[0], action[1])
        self.car.calcVelocity()
        self.car.calcPosition()

        new_x, new_y = self.car.getPosition()
        vx, vy = self.car.getVelocity()

        poss_pts = self.track.detectWall(state[0], state[1], new_x, new_y)
        for i in range(len(poss_pts)):
            pt = poss_pts[i]
            x = pt[0]
            y = pt[1]

            if [x,y] in self.track.finishPos:
                return (x, y, vx, vy)
            elif self.track.getCell(pt[0], pt[1]) == '#':
                pt = poss_pts[i-1]
                return (pt[0], pt[1], 0, 0)

        next_state = (new_x, new_y, vx, vy)
        return next_state
        
    def bestValue(self, state):
        value = float('-inf')
        actions = self.values.columns.tolist()
        # if [state[0], state[1]] in self.track.finishPos:
        #     print(state)
        for action in actions:
            new_value = self.values.at[state, action]
            if new_value > value:
                value = new_value
        return value