import pandas as pd
import numpy as np
import sys


class Data:
    def __init__(self, file):
        self.raw_data = self.read_data(file)
        self.raw_x = self.raw_data[:, 1:].astype(float)
        self.raw_y = self.raw_data[:, 0].copy()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def read_data(self, file):
        try:
            data = pd.read_csv(file, header=None, usecols=[i for i in range(1, 32)])
            data = data.to_numpy()
        except:
            print(f"error opening the file")
            exit()
        return data

    def init_data(self):
        self.one_hot_encoding()
        self.raw_data = np.concatenate((self.raw_x, self.raw_y), axis=1)

    def one_hot_encoding(self):
        self.cat_to_num()
        ret = np.zeros((len(self.raw_y), 2))
        for i, val in enumerate(self.raw_y.flat):
            ret[i, val] = 1
        self.raw_y = ret

    def cat_to_num(self):
        encode = ["M", "B"]
        for i, val in enumerate(encode):
            self.raw_y[self.raw_y == val] = i

    # def split_data(self, data, val_split, label_index=-1):
    #     np.random.shuffle(data)
    #     categories = [data[data[:, label_index] == l] for l in config.labels]

    #     ratios = [int(len(cat) * val_split // 100) for cat in categories]

    #     train = np.concatenate([cat[: -ratios[i]] for i, cat in enumerate(categories)])
    #     np.random.shuffle(train)

    #     val = np.concatenate([cat[-ratios[i] :] for i, cat in enumerate(categories)])
    #     np.random.shuffle(val)

    #     return train, val
