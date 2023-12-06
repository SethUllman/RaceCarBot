import numpy as np

# ---------------- Car ----------------
class Car:

    def __init__(self):
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)

        self.prev_position = np.zeros(2)
        self.prev_velocity = np.zeros(2)
        self.prev_acceleration = np.zeros(2)

    # ---------------- Postition ----------------
    def getPosition(self):
        return(self.position)

    def updatePosition(self, x, y):
        self.prev_position = self.position
        self.position = [x, y]
        return
    
    def calcPosition(self):
        x = self.position[0] + self.velocity[0]
        y = self.position[1] + self.velocity[1]
        self.updatePosition(x, y)

    # ---------------- Velocity ----------------
    def getVelocity(self):
        return(self.velocity)

    def updateVelocity(self, vx, vy):
        self.prev_velocity = self.velocity
        self.velocity = [vx, vy]

    def calcVelocity(self):
        vx = self.velocity[0] + self.acceleration[0]
        if vx > 5 or vx < -5: vx = self.prev_velocity[0]
        vy = self.velocity[1] + self.acceleration[1]
        if vy > 5 or vy < -5: vy = self.prev_velocity[1]
        self.updateVelocity(vx, vy)

    # ---------------- Acceleration ----------------
    def getAcceleration(self):
        return(self.acceleration)

    def updateAcceleration(self, ax, ay):
        self.acceleration = [ax, ay]
        return

    def calcAcceleration(self):
        ax = self.velocity[0] - self.prev_velocity[0]
        ay = self.velocity[1] - self.prev_velocity[1]
        self.updateAcceleration(ax, ay)
        return