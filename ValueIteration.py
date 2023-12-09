from types import new_class
import numpy as np
import pandas as pd
import random
import ast

class ValueIteration:
    def __init__(self, car, track, memory, reset):
        self.car = car
        self.track = track
        self.reset = reset  # reset car back to starting line

        # set table from saved csv file
        if memory is not None and not memory.empty:
            self.values = memory     #Q-Table
            self.values["Unnamed: 0"] = [ast.literal_eval(x) for x in self.values["Unnamed: 0"]] # convert str to tuple
            self.values = self.values.set_index("Unnamed: 0")
            col = [ast.literal_eval(x) for x in self.values.columns.to_list()] # convert str to tuple
            self.values = self.values.set_axis(col, axis=1)
            
        # initialize table values to -1.0
        else:
            indexTuples = [(row, col, x, y) for row in range(0, self.track.row) for col in range(0, self.track.col) for x in range(-5, 6) for y in range(-5, 6)]
            labels = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)] # acceleration actions
            self.values = pd.DataFrame(index=indexTuples, columns=labels, dtype=float)
            self.values.fillna(float('-1.0'), inplace=True)

    # store track positions on the finish line
    def setFinishLine(self):
        positions = self.track.finishPos
        labels = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)] # acceleration actions

        for pos in positions:
            for vx in range(-5, 6):
                for vy in range(-5, 6):
                    for a in labels:
                        self.values.at[(pos[0], pos[1], vx, vy), a] = 0.0   # set finish line positions to values of 0.0

    # run Value Iteration
    def value_iteration(self, bellmanError, discount):
        delta = float('inf')                                # difference between best value and calculated value
        delta_running_avg = [10.0, 10.0, 10.0, 10.0, 10.0]  # delta length 5 running avg 
        delta_avg = [float('inf'), float('inf')]            # holds last two delta avgs
        diff_avg = float('inf')                             # difference of the last two delta avgs
        count = 0
        self.setFinishLine()                                # initialize finish line positions
        index = 0                                           # update delta_running_avg index

        # update values until they converge
        while diff_avg > bellmanError:
            delta = 0.0

            states = self.values.index.tolist()
            actions = self.values.columns.tolist()

            for state in states:
            
                # skip calculating the state action pair's value if it's on the finish line
                if [state[0], state[1]] in self.track.finishPos:
                    continue

                v = self.bestValue(state)   # find best value among the actions
                max_value = float('-inf')

                for action in actions:
                    # skip calculating the state action pair's value if it's the wall
                    if self.track.getCell(state[0], state[1]) == '#':
                        self.values.at[state, action] = float('-inf')
                        continue

                    # Bellman equation
                    value = (self.reward(state, action) + discount*self.bestValue(self.getNextState(state, action)))
                    self.values.at[state, action] = value
                    
                    # update value if its better (car is on its way to the finish line)
                    if value > max_value:
                        max_value = value

                # calculate delta
                if self.track.getCell(state[0], state[1]) != '#':
                    V = max_value # round 2
                    if delta < abs(v - V):
                        delta = abs(v - V) # round 4

                        delta_running_avg[index] = delta

            # update delta_running_avg index
            if index < 4:
                index += 1
            else:
                index = 0            

            # calculate the delta difference for convergence
            delta_avg[1] = sum(delta_running_avg) / 5
            diff_avg = abs(delta_avg[0] - delta_avg[1])
            delta_avg[0] = delta_avg[1]

            count+=1

            # print delta info
            print(delta_running_avg)
            print("Delta:", delta, "Avg:", diff_avg)
            print("Count: ",count)

            # write table values
            self.values.to_csv('./values_tables/O_tables/values_table_O_' + str(bellmanError) + '_' + str(discount) + '.csv')

        return self.values

    # calcuate reward/cost
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
        
    # get next state from current state action pair
    def getNextState(self, state, action):
        # update state
        self.car.updatePosition(state[0], state[1])
        self.car.updateVelocity(state[2], state[3])

        # non-deterministic
        prob = random.random()
        if prob > 0.2:
            self.car.updateAcceleration(action[0], action[1])   # 80% chance of accelerating 
        else:
            self.car.updateAcceleration(0,0)                    # 20% chance of not accelerating

        # calculate next state
        self.car.calcVelocity()
        self.car.calcPosition()

        new_x, new_y = self.car.getPosition()
        vx, vy = self.car.getVelocity()

        # list of pts on the path to the next state
        poss_pts = self.track.detectWall(state[0], state[1], new_x, new_y)
        for i in range(len(poss_pts)):
            pt = poss_pts[i]
            x = pt[0]
            y = pt[1]

            # next state is on the finish line
            if [x,y] in self.track.finishPos:
                return (x, y, vx, vy)
            
            # if the position is part of the wall
            elif self.track.getCell(pt[0], pt[1]) == '#':
                # print("Previous Pt: (", state[0], state[1], ")")

                # reset car back to the starting line
                if (self.reset):
                    index = random.randrange(len(self.track.startPos))
                    start_pos = self.track.startPos[index]
                    # print("Restart: (", start_pos[0], start_pos[1], ")")
                    return (start_pos[0], start_pos[1], 0, 0)
                
                # set car to the position before hitting the wall at (0,0) velocity
                else:
                    pt = poss_pts[i-1]
                    # print("Nearest Pt: (", pt[0], pt[1], ")")
                    return (pt[0], pt[1], 0, 0)

        next_state = (new_x, new_y, vx, vy)
        return next_state
        
    # find best value from the list of actions
    def bestValue(self, state):
        value = float('-inf')
        actions = self.values.columns.tolist()

        for action in actions:
            new_value = self.values.at[state, action]

            if new_value > value:
                value = new_value
        return value
    
    # find best valued starting position & policy/path
    def bestStart(self):
        startValue = float('-inf')
        state = (0,0,0,0)
        bestAction = (0, 0)
        state_action_pairs = [] # holds path to the finish line
        actions = self.values.columns.tolist()

        # find best valued starting position
        for start in self.track.startPos:
            state = (start[0], start[1], 0, 0)
            
            for action in actions:

                if startValue < self.values.at[state, action]:
                    bestAction = action
        
        # get best next state after starting position
        best_next_state = self.getNextState(state, bestAction)

        # get moves and best policy/path to the finish line
        move_count, state_action_pairs = self.bestPolicy(best_next_state, state_action_pairs)

        print("Move Count:", move_count)
        print("Path Taken:", state_action_pairs, "\n")
        if (move_count < 100):
            self.printPath(state_action_pairs)

    # find best policy/path
    def bestPolicy(self, state, state_action_pairs):
        move_count = 1
        actions = self.values.columns.tolist()

        # find get best action and move along track until finish line is crossed
        while(True):
            bestValue = float('-inf')
            bestAction = (0, 0)

            for action in actions:

                if bestValue < self.values.at[state, action]:
                    bestValue = self.values.at[state, action]
                    bestAction = action
            
            # add best next state action pair to path
            state_action_pairs.append([state, bestAction])
            best_next_state = self.getNextState(state, bestAction)

            # end if car crosses finish line
            if [best_next_state[0], best_next_state[1]] in self.track.finishPos:
                return move_count, state_action_pairs
            
            # end if path can't be found after 100 moves
            elif move_count >= 100:
                return move_count, ["BAD"]
            
            # update state and add move
            else:
                state = best_next_state
                move_count += 1

    def printPath(self, state_action_pairs):
        for state_action in state_action_pairs:
            state, action = state_action
            self.track.setCell('C', state[0], state[1])
        print(self.track)
