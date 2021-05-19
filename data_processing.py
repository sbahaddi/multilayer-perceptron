import pandas as pd
import sys


class Data:
    def __init__(self, file):
        self.data = self.read_data(file)
        self.X = self.data.to_numpy()[:, 1:].astype(float)
        self.Y = self.data.to_numpy()[:, 0]

    def read_data(self, file):
        try:
            data = pd.read_csv(file, header=None, usecols=[i for i in range(1, 32)])
            data = data
        except:
            print(f"error opening the file")
            exit()
        return data
