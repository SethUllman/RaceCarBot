import numpy as np

# --------------- Track ---------------
class Track:

    def __init__(self):
        self.filename = ""
        self.row = 0
        self.col = 0
        self.stateSpace = 0
        self.startPos = []
        self.track = None     # holds cell values in the track as a 2D numpy array

    # populates cells into Track
    def parseTrack(self, filename):
        self.filename = filename

        file = open(filename, 'r')    # create Track by reading in txt file

        # initialize track board
        size = file.readline().replace("\n", "").split(",")
        self.row = int(size[0])
        self.col = int(size[1])
        self.track = np.empty((self.row, self.col), dtype=object)

        # assign cell values to the track
        for row in range(self.row):
            line = file.readline().replace("\n", "")  # read row of cell values
            for col in range(self.col):
                self.track[row][col] = line[col]
                if self.track[row][col] in [".", "S"]: self.stateSpace += 1
                if self.track[row][col] == "S": self.startPos.append([row, col])

    def getCell(self, row, col):
        return(self.track[row][col])

    def setCell(self, cell, row, col):
        self.track[row][col] = cell
        return

    def setTrack(self, track):
        self.track = track
        return
    
    def detectWall(self, x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while x0 != x1 or y0 != y1:
            points.append((x0, y0))
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        points.append((x0, y0))
        return points

    # prints track as one string
    def __str__(self):
        full_track = ""
        for row in range(self.row):
            for col in range(self.col):
                full_track += self.track[row][col]
            full_track += "\n"
        return full_track