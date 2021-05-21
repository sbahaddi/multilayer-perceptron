import pandas as pd
import numpy as np
import config as cfg
# np.random.seed(1337)


class Data:
    def __init__(self, file):
        self.raw_data = self.read_data(file)
        self.raw_x = self.raw_data[:, 1:].astype(float)
        self.raw_y = self.raw_data[:, 0].copy()
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.x_max = None
        self.x_min = None

    def read_data(self, file):
        try:
            data = pd.read_csv(file, header=None, usecols=[i for i in range(1, 32)])
            data = data.to_numpy()
        except:
            print(f"error opening the file")
            exit()
        return data

    def init_data(self):
        train, val = self.split_data()
        self.X_train,self.y_train = train[:,1:].astype(float), self.one_hot_encoding(train[:,0])
        self.X_val,self.y_val = val[:,1:].astype(float), self.one_hot_encoding(val[:,0])
        self.X_train = self.scale_data(self.X_train)
        self.X_val = self.scale_data(self.X_val)

    def scale_data(self, X):
        if(self.x_max is  None and self.x_min is None):
            self.x_max = X.max(axis=0)
            self.x_min = X.min(axis=0)
        return (X - self.x_min) / (self.x_max - self.x_min)

    def one_hot_encoding(self, y):
        categories = cfg.categories
        for i, val in enumerate(categories):
            y[y == val] = i
        encode = np.zeros((len(y), 2))
        for i, val in enumerate(y.flat):
            encode[i, val] = 1
        return encode

    def split_data(self):
        np.random.shuffle(self.raw_data)
        categories = [self.raw_data[self.raw_data[:, 0] == l] for l in cfg.categories]

        ratios = [int(len(cat) * cfg.split_val // 100) for cat in categories]

        train = np.concatenate([cat[: ratios[i]] for i, cat in enumerate(categories)])
        np.random.shuffle(train)

        val = np.concatenate([cat[ratios[i] :] for i, cat in enumerate(categories)])
        np.random.shuffle(val)

        return train, val
