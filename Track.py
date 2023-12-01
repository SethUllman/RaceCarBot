import numpy as np

# --------------- Track ---------------
class Track:

    def __init__(self):
        self.filename = ""
        self.row = 0
        self.col = 0
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

    def getCell(self, row, col):
        return(self.track[row][col])

    def setCell(self, cell, row, col):
        self.track[row][col] = cell
        return

    def setTrack(self, track):
        self.track = track
        return

    # prints track as one string
    def __str__(self):
        full_track = ""
        for row in range(self.row):
            for col in range(self.col):
                full_track += self.track[row][col]
            full_track += "\n"
        return full_track